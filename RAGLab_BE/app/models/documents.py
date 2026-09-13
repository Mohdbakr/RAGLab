from fastapi import UploadFile


def validate_document(file: UploadFile) -> bool:
    """Validate if the file type is supported.

    Args:
        file: The uploaded file to validate

    Returns:
        bool: True if file type is supported, False otherwise
    """
    supported_types: set[str] = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    return file.content_type in supported_types
