from argparse import ArgumentTypeError, Namespace
from contextlib import asynccontextmanager
from enum import Enum

from astropy.units import Quantity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, JsonValue

from .ETC_arguments import (
    ArgumentParserError,
    check_inputs_add_units,
    noQuitETCparser,
)
from .ETC_main import main as run_etc_main


class ETCMode(str, Enum):
    SNR = "SNR"
    EXPTIME = "EXPTIME"
    SET = "SET"


class Slit(str, Enum):
    SET = "SET"
    LOSS = "LOSS"
    RES = "RES"
    SNR = "SNR"
    AUTO = "AUTO"


class MagSystem(str, Enum):
    AB = "AB"
    VEGA = "VEGA"


class SNRRequest(BaseModel):
    """Used for request Schema, should validate data"""

    channel: str
    wrange: list[float]  # positive
    etc_mode: ETCMode
    etc_fixed: float  # positive

    binspect: int = 1
    binspat: int = 1
    noslicer: bool = False
    fastSNR: int | None = None
    plotSNR: bool = False
    plotslit: bool = False
    timer: bool = False
    hires: bool = False
    hires_solve: bool = False

    slit: Slit
    slitwidth: float | None = None
    seeing_fwhm: float  # positive
    seeing_pivot: float  # positive
    airmass: float
    skymag: float
    skyfilter: str = "R"

    mag: float
    magsystem: MagSystem
    magfilter: str

    model: list[str] = ["constant"]
    z: float = 0.0  # positive
    e_bv: float = 0.0

    extmodel: str = "mwavg"
    extended: int | None = None


def build_args(req: SNRRequest) -> Namespace:
    """Translate a validated request into the Namespace shape ETC_main.main() expects."""
    slit = (
        [req.slit.value] if req.slitwidth is None else [req.slit.value, req.slitwidth]
    )

    return Namespace(
        channel=req.channel,
        wrange=list(req.wrange),
        ETCmode=req.etc_mode.value,
        ETCfixed=req.etc_fixed,
        slit=slit,
        seeing=[req.seeing_fwhm, req.seeing_pivot],
        airmass=req.airmass,
        skymag=req.skymag,
        skyfilter=req.skyfilter,
        mag=req.mag,
        magsystem=req.magsystem.value,
        magfilter=req.magfilter,
        model=req.model,
        z=req.z,
        E_BV=req.e_bv,
        extmodel=req.extmodel,
        extended=req.extended,
        binspect=req.binspect,
        binspat=req.binspat,
        noslicer=req.noslicer,
        fastSNR=req.fastSNR,
        hires=req.hires,
        hires_solve=req.hires_solve,
        # Plotting/timing are CLI-only concerns; always off for the API path
        plotSNR=False,
        plotslit=False,
        timer=False,
    )


def serialize_result(result: object) -> JsonValue:
    if isinstance(result, Quantity):
        value = result.value
        if hasattr(value, "tolist"):
            value = value.tolist()
        return {
            "value": value,
            "unit": str(result.unit),
        }
    if isinstance(result, dict):
        return {str(key): serialize_result(value) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        return [serialize_result(value) for value in result]
    if result is None or isinstance(result, (bool, int, float, str)):
        return result
    return str(result)


def compute_snr(req: SNRRequest) -> JsonValue:
    args = build_args(req)
    check_inputs_add_units(args)
    result = run_etc_main(args, quiet=True)
    return serialize_result(result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    # Make parser.error() raise instead of sys.exit()-ing the whole worker
    noQuitETCparser()

    # maybe import slow modules

    yield


app = FastAPI(title="ETC Service", lifespan=lifespan)

@app.post("/etc")
def run_etc(req: SNRRequest) -> JsonValue:
    try:
        return compute_snr(req)
    except (ArgumentParserError, ArgumentTypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))


# sample input
# {
#   "channel": "R",
#   "wrange": [600, 750],
#   "etc_mode": "EXPTIME",
#   "etc_fixed": 100,
#   "binspect": 1,
#   "binspat": 1,
#   "noslicer": false,
#   "fastSNR": null,
#   "plotSNR": false,
#   "plotslit": false,
#   "timer": false,
#   "hires": false,
#   "hires_solve": false,
#   "slit": "AUTO",
#   "slitwidth": null,
#   "seeing_fwhm": 1.0,
#   "seeing_pivot": 5000,
#   "airmass": 1.2,
#   "skymag": 20.0,
#   "skyfilter": "R",
#   "mag": 20.0,
#   "magsystem": "AB",
#   "magfilter": "R",
#   "model": ["constant"],
#   "z": 0.0,
#   "e_bv": 0.0,
#   "extmodel": "mwavg",
#   "extended": null
# }


def main():
    import uvicorn
    uvicorn.run("etc_microservice.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
