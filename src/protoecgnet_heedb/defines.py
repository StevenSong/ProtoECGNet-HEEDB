import os
from typing import Literal

SIM_MAX = 1e6
# label value for a (sample, label) whose truth can't be determined; excluded from
# losses and metrics. only some label sources produce it, e.g. HEEDB audited_physician
UNKNOWN_LABEL = -1

STAGE_T = Literal[
    "learn-prototypes-supervised",
    "project-prototypes-supervised",
    "compute-embeddings",
    "train-classifier",
    "train-fusion-classifier",
    "compute-fusion-embeddings",
]

SPLIT_T = Literal["train", "val", "test"]
PROT_T = Literal["global", "partial"]
CONV_T = Literal["1D", "2D"]
BACKBONE_T = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

STANDARD_LEAD_ORDER = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

# same default dir as ProtoSSL: cache keys hash the inputs and loader code, so the two
# packages share entries only when those match, and the existing caches stay usable
CACHE_DIR = os.environ.get(
    "CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache/protossl_cache")
)

# HEEDB per-lead stats derived over train set (data before 2021) at 250 or 500 Hz source freq
# 250 Hz samples were weighted equally to 500 Hz samples (see ProtoSSL's scripts/pretrain/_compute_heedb_normalizations.py)
HEEDB_MGB_LOWERS = [
    -1.0119186174765404,
    -1.4390830926038203,
    -1.7198238541919237,
    -1.1813091118326642,
    -0.9979542501410862,
    -1.4866640276672654,
    -1.7597675495680554,
    -2.5090881325221543,
    -2.3733176732190997,
    -1.8311383205364702,
    -1.5223512471635987,
    -1.6062121084788448,
]
HEEDB_MGB_UPPERS = [
    1.3460418197872064,
    1.5057521770008284,
    1.3717106293294237,
    0.9382710634263587,
    1.2941140484114941,
    1.3165757358737804,
    1.193413322647441,
    1.714575963086197,
    1.7340191625060277,
    1.9032238785473585,
    1.8878508873821698,
    1.7253424693450583,
]
HEEDB_MGB_CLIPPED_MEANS = [
    -0.0006269784908739345,
    -0.0032141328048856037,
    -0.0025533607205344565,
    0.0019646277729164162,
    0.0010130726248037739,
    -0.002853690357906745,
    -0.005724228465493457,
    -0.004851620104470237,
    -0.003503455167473047,
    -0.0013349983274631835,
    -0.001980557557397161,
    -0.00492753875377437,
]
HEEDB_MGB_CLIPPED_STDS = [
    0.15993755158963452,
    0.1935362312377656,
    0.19184339812753923,
    0.1503154757829258,
    0.1483126573949368,
    0.17524810046198744,
    0.19213824105393,
    0.2794094184989707,
    0.2748612949764635,
    0.24639481851816147,
    0.22539578223901716,
    0.2110454711343999,
]

HEEDB_EUH_LOWERS = [
    -0.7038517368573582,
    -1.1687722073608027,
    -1.5766076874360642,
    -1.1474148774929622,
    -0.6107907486696409,
    -0.7880775391316841,
    -1.7644907463452804,
    -2.4848630681527233,
    -2.317365931668375,
    -1.7669268872193773,
    -1.3187983709915314,
    -1.0608492130819662,
]
HEEDB_EUH_UPPERS = [
    1.2977676267366527,
    1.4483912206315044,
    1.2221323710167684,
    0.6722268319333242,
    0.7880201664232516,
    0.6109128117841898,
    1.0197808717988628,
    1.3871085592096128,
    1.5421727640098344,
    1.843899560056582,
    1.8834247685566945,
    1.7067736639285025,
]
HEEDB_EUH_CLIPPED_MEANS = [
    -0.00333641190429124,
    -0.0015139232433631325,
    0.0018329751341113363,
    0.002425127093554142,
    -0.0009166117438822255,
    0.0009166761347320038,
    -0.00442523197358341,
    -0.004174428258389581,
    -0.00363840053302034,
    -0.0032306875760891867,
    -0.0031563845364006465,
    -0.0031243030602937636,
]
HEEDB_EUH_CLIPPED_STDS = [
    0.1484186659504811,
    0.17659226851748097,
    0.17035433460633997,
    0.13951416925077822,
    0.08517362507419514,
    0.08517502277442,
    0.18804115732392176,
    0.26205856315682696,
    0.25674355934896154,
    0.23260461704801536,
    0.2119686514178204,
    0.18862748143524874,
]

HEEDB_TARGETS = {
    "ANTERIOR INFARCT": [740],
    "ATRIAL FIBRILLATION": [161],
    "ATRIAL FLUTTER": [162, 273, 288],
    "ATRIAL-PACED RHYTHM": [295],
    "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK": [445],
    "INFERIOR INFARCT": [780],
    "LATERAL INFARCT": [760],
    "LEFT BUNDLE BRANCH BLOCK": [460],
    "NORMAL SINUS RHYTHM": [22],
    "PREMATURE ATRIAL COMPLEXES": [222],
    "PREMATURE VENTRICULAR COMPLEXES": [231],
    "RIGHT AXIS DEVIATION": [383],
    "RIGHT BUNDLE BRANCH BLOCK": [440],
    "SINUS BRADYCARDIA": [21],
    "SINUS RHYTHM": [19],
    "SINUS TACHYCARDIA": [23],
    "VENTRICULAR TACHYCARDIA": [248, 272, 277],
    "VENTRICULAR-PACED RHYTHM": [296],
    "WITH 1ST DEGREE AV BLOCK": [101],
    "WITH SINUS ARRHYTHMIA": [251],
}


def env_flag(name: str) -> bool:
    # unlike a plain truthiness check, "0" and "false" turn the flag off
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    raise ValueError(f"Unrecognized value for {name}: {value!r}")


HEEDB_LEAD_ORDER = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
