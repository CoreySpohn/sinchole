"""Shared fixtures for sinchole tests."""

import gc
import importlib.util
import os

import jax
import jax.numpy as jnp
import pytest
from hypothesis import settings

import sinchole as sc

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def clear_compile_cache():
    yield
    jax.clear_caches()
    gc.collect()


def analytic(source, asymmetric=False, shape=(8, 6)):
    rows, columns = jnp.indices(shape, dtype=jnp.float64)
    xpix = (columns - (shape[1] - 1) / 2) * 0.25
    ypix = (rows - (shape[0] - 1) / 2) * 0.25
    x = jnp.cos(2 * jnp.pi * 0.16 * (source[0] - xpix))
    y = jnp.cos(2 * jnp.pi * 0.12 * (source[1] - ypix))
    image = 2 + 0.4 * x + 0.3 * y + 0.1 * x * y
    if asymmetric:
        image = image + 0.3 * jnp.sin(2 * jnp.pi * 0.1 * source[0]) + 0.04 * xpix
    return image


def delivery(quadrant=False, asymmetric=False, half=20, shape=(8, 6)):
    count = half + 1 if quadrant else 2 * half + 1
    origin = 0.0 if quadrant else -half * 0.25
    lattice = sc.Lattice(jnp.full(2, origin), jnp.full(2, 0.25), (count, count))
    iy, ix = jnp.indices(lattice.shape_yx, dtype=jnp.int32)
    indices = jnp.stack((ix.ravel(), iy.ravel()), axis=1)
    offsets = lattice.origin_xy_lod + indices * lattice.spacing_xy_lod
    images = jax.vmap(lambda q: analytic(q, asymmetric, shape))(offsets)[None]
    pixels = sc.PixelGrid(
        shape,
        jnp.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2]),
        jnp.eye(2) * 0.25,
        "pixel_integral",
    )
    products = sc.Products(
        sc.Frame(jnp.array(500.0), jnp.array(6.0), "entrance_diameter"),
        pixels,
        sc.Normalization(
            "incident_photon", jnp.array([1.0]), jnp.array([jnp.nan]), (), "units"
        ),
        "monochromatic",
        jnp.array([500.0]),
        jnp.array([[500.0, 500.0]]),
        jnp.array([500.0]),
        jnp.array(jnp.nan),
        jnp.ones((1, 2)),
        jnp.ones(1, dtype=jnp.int32),
        ("support",),
        sc.Premises(
            jnp.ones(6, dtype=bool), jnp.ones(6, dtype=jnp.int32), ("truth",) * 6
        ),
        "optics",
        "mono",
        "synthetic",
    )
    reflection = sc.Reflection(
        "two_axis_pixel_flip_v1" if quadrant else "none",
        jnp.full(2, quadrant),
        "declaration",
        "argument",
        jnp.array(1, dtype=jnp.int32),
        "checks",
        jnp.array([False]),
    )
    domain = sc.Domain(
        "box", "closed", jnp.array([[-1.0, -1.0], [1.0, 1.0]]), jnp.zeros(2)
    )
    mirrors = None
    if quadrant:
        signed = jnp.array([[-1, 1], [1, -1], [-1, -1]], dtype=jnp.int32)
        positions = signed * 0.25
        mirrors = sc.MirrorCheckSet(
            positions,
            signed,
            jax.vmap(lambda q: analytic(q, asymmetric, shape))(positions),
            jnp.zeros(3, dtype=jnp.int32),
            "independent",
        )
    return dict(
        intensity_nodes=images,
        offsets_xy_lod=offsets,
        declared_lattice=lattice,
        indices_xy=indices,
        products=products,
        reflection=reflection,
        domain=domain,
        memory=sc.MemoryBudget(100_000_000, 10_000_000, 10_000_000, 400_000_000),
        mirror_checks=mirrors,
    )


def make_reader(data):
    admission = sc.admit_grid(**data)
    grid = admission.require()
    return sc.Interpolator(
        grid,
        data["domain"],
        sc.KernelProfile(),
        sc.ReadPolicy(),
        sc.Tiling(pixel_tile=13, query_chunk=2),
        data["memory"],
    )


@pytest.fixture(scope="module")
def full_delivery():
    return delivery()


@pytest.fixture(scope="module")
def full_reader(full_delivery):
    return make_reader(full_delivery)


@pytest.fixture(scope="module")
def quadrant_delivery():
    return delivery(quadrant=True)


@pytest.fixture(scope="module")
def quadrant_reader(quadrant_delivery):
    return make_reader(quadrant_delivery)
