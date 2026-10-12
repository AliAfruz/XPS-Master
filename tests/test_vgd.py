import struct

import numpy as np

from xps_afruz.io import vgd


class _FakeOleDocument:
    def __init__(self, streams: dict[str, bytes]) -> None:
        self.streams = streams

    def listdir(self, *, streams: bool, storages: bool):
        return [[name] for name in self.streams]

    def openstream(self, name):
        stream_name = name[-1] if isinstance(name, list) else name

        class _Stream:
            def __init__(self, data: bytes) -> None:
                self.data = data

            def read(self) -> bytes:
                return self.data

        return _Stream(self.streams[stream_name])

    def close(self) -> None:
        pass


class _FakeOleModule:
    OleFileIO = None

    @staticmethod
    def isOleFile(path: str) -> bool:
        return True


def test_vgd_import_converts_kinetic_to_binding_energy(monkeypatch, tmp_path) -> None:
    data_axes = bytearray(40)
    struct.pack_into("<i", data_axes, 12, 3)  # zero-based: four points
    struct.pack_into("<i", data_axes, 28, 0)  # zero-based: one spectrum
    space_axes = bytearray(46)
    struct.pack_into("<d", space_axes, 30, 100.0)
    struct.pack_into("<d", space_axes, 38, 0.5)
    property_stream = struct.pack("<4f", 0.0, 1486.68, 0.0, 0.0)
    streams = {
        "VGData": np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8").tobytes(),
        "VGDataAxes": bytes(data_axes),
        "VGSpaceAxes": bytes(space_axes),
        "\x05CustomProperties": property_stream,
    }
    fake_module = _FakeOleModule()
    fake_module.OleFileIO = lambda path: _FakeOleDocument(streams)
    monkeypatch.setattr(vgd, "olefile", fake_module)
    source = tmp_path / "sample.VGD"
    source.write_bytes(b"fake")

    spectrum = vgd.read_vgd_spectrum(source)

    assert np.allclose(spectrum.energy, [1385.18, 1385.68, 1386.18, 1386.68])
    assert np.allclose(spectrum.intensity, [40.0, 30.0, 20.0, 10.0])
    assert spectrum.metadata["format"] == "Thermo/VG VGD"
    assert spectrum.energy_scale.value == "binding"
    assert spectrum.original_energy_scale.value == "kinetic"
    assert np.isclose(spectrum.photon_energy_eV, 1486.68)
