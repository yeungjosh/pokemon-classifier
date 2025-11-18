import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, mock_open
from io import BytesIO
import aiohttp
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_learner():
    """Mock fastai learner"""
    mock = Mock()
    mock.predict = Mock(return_value=('Pikachu', None, None))
    return mock


@pytest.fixture
def client(mock_learner):
    """Test client with mocked learner"""
    with patch('app.server.learn', mock_learner):
        from app import server
        with TestClient(server.app) as client:
            yield client


class TestDownloadFile:
    """Test file download functionality"""

    @pytest.mark.asyncio
    async def test_download_file_creates_new(self, tmp_path):
        """Should download file if doesn't exist"""
        from app.server import download_file

        dest = tmp_path / "test.pkl"
        test_data = b"test data"

        mock_response = AsyncMock()
        mock_response.read = AsyncMock(return_value=test_data)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            await download_file("http://test.com/file", dest)

        assert dest.exists()
        assert dest.read_bytes() == test_data

    @pytest.mark.asyncio
    async def test_download_file_skips_existing(self, tmp_path):
        """Should skip download if file exists"""
        from app.server import download_file

        dest = tmp_path / "existing.pkl"
        dest.write_bytes(b"existing data")

        with patch('aiohttp.ClientSession') as mock_session:
            await download_file("http://test.com/file", dest)

        mock_session.assert_not_called()
        assert dest.read_bytes() == b"existing data"


class TestEndpoints:
    """Test HTTP endpoints"""

    def test_homepage_returns_html(self, client):
        """GET / should return HTML"""
        response = client.get('/')
        assert response.status_code == 200
        assert 'text/html' in response.headers['content-type']

    def test_analyze_with_valid_image(self, client, mock_learner):
        """POST /analyze should classify image"""
        img_data = BytesIO(b"fake image data")
        img_data.name = "test.jpg"

        with patch('app.server.open_image') as mock_open_image:
            mock_img = Mock()
            mock_open_image.return_value = mock_img

            response = client.post(
                '/analyze',
                files={'file': ('test.jpg', img_data, 'image/jpeg')}
            )

        assert response.status_code == 200
        assert 'result' in response.json()
        assert response.json()['result'] == 'Pikachu'
        mock_learner.predict.assert_called_once_with(mock_img)

    def test_analyze_without_file(self, client):
        """POST /analyze without file should fail"""
        response = client.post('/analyze')
        assert response.status_code in [400, 422]

    def test_cors_headers(self, client):
        """Should have CORS middleware configured"""
        response = client.options('/analyze', headers={
            'Origin': 'http://example.com',
            'Access-Control-Request-Method': 'POST'
        })
        assert 'access-control-allow-origin' in response.headers


class TestSetupLearner:
    """Test learner setup"""

    @pytest.mark.asyncio
    async def test_setup_learner_downloads_and_loads(self, tmp_path):
        """Should download model and load learner"""
        from app.server import setup_learner

        mock_learner = Mock()

        with patch('app.server.download_file', new_callable=AsyncMock) as mock_dl, \
             patch('app.server.load_learner', return_value=mock_learner) as mock_load, \
             patch('app.server.path', tmp_path):

            result = await setup_learner()

        mock_dl.assert_called_once()
        mock_load.assert_called_once()
        assert result == mock_learner

    @pytest.mark.asyncio
    async def test_setup_learner_cpu_error_message(self, tmp_path):
        """Should give helpful message for CPU-only errors"""
        from app.server import setup_learner

        with patch('app.server.download_file', new_callable=AsyncMock), \
             patch('app.server.load_learner') as mock_load, \
             patch('app.server.path', tmp_path):

            mock_load.side_effect = RuntimeError("CPU-only machine detected")

            with pytest.raises(RuntimeError) as exc:
                await setup_learner()

            assert "old version of fastai" in str(exc.value)

    @pytest.mark.asyncio
    async def test_setup_learner_reraises_other_errors(self, tmp_path):
        """Should re-raise non-CPU errors"""
        from app.server import setup_learner

        with patch('app.server.download_file', new_callable=AsyncMock), \
             patch('app.server.load_learner') as mock_load, \
             patch('app.server.path', tmp_path):

            mock_load.side_effect = RuntimeError("Other error")

            with pytest.raises(RuntimeError) as exc:
                await setup_learner()

            assert "Other error" in str(exc.value)
