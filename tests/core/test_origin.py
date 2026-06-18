from app.core.origin import is_localhost_origin, normalize_origin


class TestNormalizeOrigin:
    def test_lowercases_host(self):
        assert normalize_origin("https://App.Example.COM") == "https://app.example.com"

    def test_strips_trailing_slash_and_path(self):
        assert normalize_origin("https://app.example.com/") == "https://app.example.com"
        assert normalize_origin("https://app.example.com/some/path") == "https://app.example.com"

    def test_drops_default_https_port(self):
        assert normalize_origin("https://app.example.com:443") == "https://app.example.com"

    def test_keeps_non_default_port(self):
        assert normalize_origin("https://app.example.com:8443") == "https://app.example.com:8443"

    def test_combination(self):
        assert normalize_origin("HTTPS://App.Example.com:443/") == "https://app.example.com"


class TestIsLocalhostOrigin:
    def test_localhost_any_port(self):
        assert is_localhost_origin("http://localhost:5173") is True
        assert is_localhost_origin("http://localhost") is True

    def test_non_localhost(self):
        assert is_localhost_origin("https://app.example.com") is False
