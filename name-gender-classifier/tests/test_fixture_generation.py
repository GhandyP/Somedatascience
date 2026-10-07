import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "regenerate_fixture.py"
spec = importlib.util.spec_from_file_location("regenerate_fixture", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_select_and_serialize_fixture_bytes(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(
        b'"year","name","percent","sex"\n'
        b'1880,"First, Name",0.1,"boy"\n'
        b'1881,"Second",0.2,"girl"\n'
        b'1880,"Third",0.3,"boy"\n'
        b'1880,"Fourth",0.4,"girl"\n'
        b'1880,"Fifth",0.5,"boy"\n'
    )
    assert generator.generate_bytes(source, per_group=2) == (
        b'"year","name","percent","sex"\n'
        b'1880,"First, Name",0.1,"boy"\n'
        b'1881,"Second",0.2,"girl"\n'
        b'1880,"Third",0.3,"boy"\n'
        b'1880,"Fourth",0.4,"girl"\n'
    )
