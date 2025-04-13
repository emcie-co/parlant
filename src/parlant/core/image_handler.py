# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import httpx
import mimetypes
import os
import base64
from uuid import uuid4
from typing import Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")



class ImageUploadError(Exception):
    """Exception raised when an image upload fails."""
    
    def __init__(self, message: str) -> None:
        super().__init__(f"Error uploading image to CDN: {message}")


def get_mime_type(file_path: str) -> str:
    """Get MIME type from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type of the file
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def get_mime_type_from_base64(base64_image: str) -> str:
    """Extract MIME type from a base64 data URL.
    
    Args:
        base64_image: Base64 encoded image with data URL prefix
        
    Returns:
        MIME type of the image
    """
    return base64_image.split(",")[0].split(":")[1].split(";")[0]


async def upload_base64_image_to_cdn(
    base64_image: str, 
    file_name: Optional[str] = None, 
    mime_type: Optional[str] = None
) -> str:
    """
    Asynchronously uploads a base64 encoded image to Supabase storage.
    
    Args:
        base64_image: Base64 encoded image string, with or without data URL prefix
        file_name: Optional custom filename to use, will generate UUID if not provided
        mime_type: Optional mime type, will be detected if not provided
        
    Returns:
        Public URL of the uploaded file
        
    Raises:
        ImageUploadError: If the upload fails
    """
    try:
        if "base64," in base64_image:
            # If base64 string includes the data URL prefix
            mime_type = get_mime_type_from_base64(base64_image)
            file_extension = mime_type.split("/")[1]
            file_data = base64_image.split(",")[1]
            file_data = base64.b64decode(file_data)
            
            if file_name is None:
                file_name = f"{uuid4()}.{file_extension}"
        else:
            # Handle raw base64 string without prefix
            file_data = base64.b64decode(base64_image)
            
            if file_name is None:
                file_name = f"{uuid4()}.png"  # Default to png if unknown
            
            if mime_type is None:
                mime_type = get_mime_type(file_name)
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": mime_type or "application/octet-stream",
            "x-upsert": "true"  # Optional: overwrite existing file
        }

        url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{file_name}"
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.put(url, headers=headers, content=file_data)

            if response.status_code == 200:
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_name}"
                return public_url
            else:
                raise ImageUploadError(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        raise ImageUploadError(str(e)) 