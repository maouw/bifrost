"""Utility functions for downloading files from URLs."""

import datetime
import email
import os
import urllib.parse

import requests
from tqdm import tqdm


def download_file(  # noqa: PLR0913
    url: str,
    filename: str | os.PathLike | None = None,
    destination_dir: str | os.PathLike = ".",
    overwrite: bool = False,
    chunk_size: int = 8192,
    user_agent: str | None = "curl/7.81.0",
) -> None | str:
    """Download a file from a URL to a specified directory.

    Args:
        url: URL of the file to download.
        filename: Optional name for the downloaded file. If not provided, it will be derived from the URL or current timestamp.
        destination_dir: Directory where the file will be saved. Defaults to the current directory.
        overwrite: If True, overwrite the file if it already exists. Defaults to False.
        chunk_size: Size of chunks to read from the response stream. Defaults to 8192 bytes.
        user_agent: User-Agent string to use for the request. Defaults to "curl/7.81.0".

    Returns:
        str: The path to the downloaded file if successful, otherwise None.

    Raises:
        FileExistsError: If the file already exists and overwrite is False.
    """
    headers = requests.utils.default_headers()
    if user_agent:
        headers["User-Agent"] = user_agent
    with requests.get(url, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        if not filename and "Content-Disposition" in resp.headers:
            filename = email.message_from_string("Content-Disposition: " + resp.headers["Content-Disposition"]).get_filename()
        if not filename:
            filename = os.path.basename(os.path.normpath(urllib.parse.urlparse(resp.url).path))
        if not filename:
            filename = os.path.basename(os.path.normpath(urllib.parse.urlparse(url).path))
        if not filename:
            filename = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ".download"
        if not os.path.isabs(filename):
            filename = os.path.join(destination_dir, filename)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
        filename = os.path.normpath(filename)
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Use overwrite=True to overwrite it.")
        with tqdm.wrapattr(open(filename, "wb"), "write", total=int(resp.headers.get("Content-Length", 0)), desc=os.path.basename(filename)) as fout:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fout.write(chunk)
    return filename
