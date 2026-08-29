from src.prepare_data.clean_directories import clean_directories


def _touch(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "payload.txt").write_text("x")


class TestCleanDirectories:
    def test_deletes_matching_top_level_dirs(self, tmp_path):
        _touch(tmp_path / "swde")
        _touch(tmp_path / "gml")
        _touch(tmp_path / "keep")

        deleted = clean_directories(tmp_path, ["swde", "gml"])

        assert deleted == 2
        assert not (tmp_path / "swde").exists()
        assert not (tmp_path / "gml").exists()
        assert (tmp_path / "keep").exists()

    def test_match_is_case_insensitive(self, tmp_path):
        # Real deliveries ship the same logical folder as SWDE / Swde / swde depending on
        # the county's export tooling — the match must not care about case.
        _touch(tmp_path / "SWDE")
        _touch(tmp_path / "Gml")

        deleted = clean_directories(tmp_path, ["swde", "GML"])

        assert deleted == 2
        assert not (tmp_path / "SWDE").exists()
        assert not (tmp_path / "Gml").exists()

    def test_deletes_nested_dirs_at_any_depth(self, tmp_path):
        _touch(tmp_path / "county" / "unit" / "shp")

        deleted = clean_directories(tmp_path, ["shp"])

        assert deleted == 1
        assert not (tmp_path / "county" / "unit" / "shp").exists()
        assert (tmp_path / "county" / "unit").exists()

    def test_pruned_dir_is_not_descended_into(self, tmp_path):
        # A matching dir is removed AND pruned from the walk, so a same-named child inside it
        # is not counted separately — it went away with its parent.
        _touch(tmp_path / "swde" / "swde")

        deleted = clean_directories(tmp_path, ["swde"])

        assert deleted == 1
        assert not (tmp_path / "swde").exists()

    def test_returns_zero_when_nothing_matches(self, tmp_path):
        _touch(tmp_path / "keep")

        assert clean_directories(tmp_path, ["swde"]) == 0
        assert (tmp_path / "keep").exists()

    def test_missing_base_dir_returns_zero(self, tmp_path):
        assert clean_directories(tmp_path / "does_not_exist", ["swde"]) == 0
