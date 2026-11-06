"""A toy Lyot coronagraph for the sinchole documentation.

The model is deliberately small: a circular entrance pupil of width D, a hard
occulting disk in the first focal plane, a circular Lyot stop and a fixed
detector, connected by matrix Fourier transforms. Every operation after the
source tilt is linear and independent of the source position, which is the
property the documentation pages rely on. It is a teaching model, not a
design: use a physical-optics package for real coronagraphs.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

import sinchole as sc

jax.config.update("jax_enable_x64", True)


def _axis(count, step):
    """Return centered sample positions with the given step."""
    return (jnp.arange(count) - (count - 1) / 2) * step


@dataclass(frozen=True)
class ToyCoronagraph:
    """Planet intensity images of a toy Lyot coronagraph.

    Attributes:
        n_pupil: Pupil samples across the entrance diameter D.
        occulter_lod: Occulting disk radius in lambda/D.
        lyot: Lyot stop diameter as a fraction of D.
        n_pix: Detector pixels per side.
        pixscale_lod: Detector pixel pitch in lambda/D.
        pupil_shift: Decenter (dx, dy) of the entrance pupil, in units of D.
            A shift along one axis keeps the intensity mirror symmetric (see
            the quadrant-symmetry page); a diagonal shift breaks it.
    """

    n_pupil: int = 48
    occulter_lod: float = 2.5
    lyot: float = 0.8
    n_pix: int = 96
    pixscale_lod: float = 0.25
    pupil_shift: tuple[float, float] = (0.0, 0.0)

    @property
    def pupil_coords(self):
        """Pupil sample coordinates (u, v) in units of D."""
        u = _axis(self.n_pupil, 1.0 / self.n_pupil)
        return jnp.meshgrid(u, u)

    @property
    def pupil(self):
        """Entrance pupil transmission, a disk of diameter D."""
        uu, vv = self.pupil_coords
        dx, dy = self.pupil_shift
        return ((uu - dx) ** 2 + (vv - dy) ** 2 <= 0.25).astype(jnp.float64)

    @property
    def detector_x(self):
        """Detector pixel-center coordinates along one axis, in lambda/D."""
        return _axis(self.n_pix, self.pixscale_lod)

    def phase_ramp(self, source_xy):
        """Return the pupil phase, in radians, that a source tilt imposes."""
        uu, vv = self.pupil_coords
        return 2 * jnp.pi * (uu * source_xy[0] + vv * source_xy[1])

    def field(self, source_xy):
        """Complex detector field for one source position in lambda/D."""
        uu, vv = self.pupil_coords
        u = uu[0]
        stop = (uu**2 + vv**2 <= (self.lyot / 2) ** 2).astype(jnp.float64)
        n_occ = 40
        xi = _axis(n_occ, 2 * self.occulter_lod / n_occ)
        xx, yy = jnp.meshgrid(xi, xi)
        occulter = (xx**2 + yy**2 <= self.occulter_lod**2).astype(jnp.float64)
        du = 1.0 / self.n_pupil
        to_focal = jnp.exp(-2j * jnp.pi * jnp.outer(xi, u)) * du
        to_lyot = jnp.exp(2j * jnp.pi * jnp.outer(u, xi)) * (xi[1] - xi[0])
        to_detector = jnp.exp(-2j * jnp.pi * jnp.outer(self.detector_x, u)) * du
        tilted = self.pupil * jnp.exp(1j * self.phase_ramp(source_xy))
        focal = to_focal @ tilted @ to_focal.T
        lyot_field = tilted - to_lyot @ (occulter * focal) @ to_lyot.T
        return to_detector @ (stop * lyot_field) @ to_detector.T

    def psf(self, source_xy):
        """Planet intensity image(s) for source positions [..., 2] in lambda/D.

        Images are point samples of intensity at pixel centers, normalized
        by the peak of the unocculted, unstopped image of the full pupil.
        """
        source_xy = jnp.asarray(source_xy, dtype=jnp.float64)
        peak = (jnp.sum(self.pupil) / self.n_pupil**2) ** 2

        def one(source):
            return jnp.abs(self.field(source)) ** 2 / peak

        flat = source_xy.reshape(-1, 2)
        images = jax.jit(jax.vmap(one))(flat)
        return images.reshape(*source_xy.shape[:-1], self.n_pix, self.n_pix)


def products_for(toy, *, identity="toy-lyot"):
    """Return the sinchole product declaration for a toy coronagraph.

    The declarations state what is true of the toy: a monochromatic product,
    point samples of intensity, an entrance support of width D in both axes,
    and fixed linear optics after the source tilt.
    """
    center = (toy.n_pix - 1) / 2
    pixels = sc.PixelGrid(
        (toy.n_pix, toy.n_pix),
        jnp.array([center, center]),
        jnp.eye(2) * toy.pixscale_lod,
        "point_density",
    )
    return sc.Products(
        frame=sc.Frame(jnp.array(500.0), jnp.array(6.0), "entrance_diameter"),
        pixels=pixels,
        normalization=sc.Normalization(
            "incident_photon",
            jnp.array([1.0]),
            jnp.array([jnp.nan]),
            (),
            "toy-unocculted-peak",
        ),
        kind="monochromatic",
        wavelength_nm=jnp.array([500.0]),
        limits_nm=jnp.array([[500.0, 500.0]]),
        sampling_min_nm=jnp.array([500.0]),
        campaign_sampling_min_nm=jnp.array(jnp.nan),
        support_extent_xy=jnp.ones((1, 2)),
        support_evidence=jnp.array([1], dtype=jnp.int32),
        support_source_ids=("toy-circular-pupil",),
        premises=sc.Premises(
            jnp.ones(6, dtype=bool),
            jnp.ones(6, dtype=jnp.int32),
            ("toy-linear-optics",) * 6,
        ),
        optical_id="toy",
        spectral_weight_id="mono",
        identity=identity,
    )


def quadrant_delivery(toy=None, *, spacing=0.38, nodes=31, domain_half=5.0):
    """Build the keyword arguments of ``sinchole.admit_grid`` for a quadrant grid.

    Args:
        toy: The coronagraph to propagate; the default toy if omitted.
        spacing: Node spacing in lambda/D, the same on both axes.
        nodes: Nodes per axis, indices 0 through nodes - 1.
        domain_half: Half-width of the square served box, in lambda/D.

    Returns:
        A dict ready for ``sinchole.admit_grid(**delivery)``.
    """
    toy = ToyCoronagraph() if toy is None else toy
    lattice = sc.Lattice(jnp.zeros(2), jnp.full(2, spacing), (nodes, nodes))
    iy, ix = jnp.indices(lattice.shape_yx, dtype=jnp.int32)
    indices = jnp.stack((ix.ravel(), iy.ravel()), axis=1)
    offsets = indices * spacing
    signed = jnp.array([[-3, 2], [3, -2], [-3, -2]], dtype=jnp.int32)
    mirrors = sc.MirrorCheckSet(
        signed * spacing,
        signed,
        toy.psf(signed * spacing),
        jnp.zeros(3, dtype=jnp.int32),
        "toy-direct-propagation",
    )
    return dict(
        intensity_nodes=toy.psf(offsets)[None],
        offsets_xy_lod=offsets,
        declared_lattice=lattice,
        indices_xy=indices,
        products=products_for(toy),
        reflection=sc.Reflection(
            "two_axis_pixel_flip_v1",
            jnp.array([True, True]),
            "toy-symmetry-declaration",
            "circular pupil, occulter and stop",
            jnp.array(1, dtype=jnp.int32),
            "toy-mirror-checks",
            jnp.array([False]),
        ),
        domain=sc.Domain(
            "box",
            "closed",
            jnp.array([[-domain_half, -domain_half], [domain_half, domain_half]]),
            jnp.zeros(2),
        ),
        memory=sc.MemoryBudget(2_000_000_000, 400_000_000, 400_000_000, 4_000_000_000),
        mirror_checks=mirrors,
    )


def reader_for(delivery):
    """Admit a delivery and return an ``Interpolator`` over its domain."""
    grid = sc.admit_grid(**delivery).require()
    return sc.Interpolator(
        grid,
        delivery["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=1024),
        delivery["memory"],
    )
