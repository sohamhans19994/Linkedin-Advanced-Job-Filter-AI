import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO



class DriveUtils:
    def __init__(self,config):
        self.gdrive_scope=["https://www.googleapis.com/auth/drive"]
        self.config = config
        self.service = None
    
    def get_gdrive_service(self):
        creds = None
        if os.path.exists(self.config['GDRIVE_TOKEN_PATH']):
            creds = Credentials.from_authorized_user_file(self.config['GDRIVE_TOKEN_PATH'], self.gdrive_scope)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.config['GDRIVE_API_CREDS_PATH'], self.gdrive_scope)
                creds = flow.run_local_server(port=0)
            with open(self.config['GDRIVE_TOKEN_PATH'], "w") as token:
                token.write(creds.to_json())
        self.service = build("drive", "v3", credentials=creds)
    
    
    def create_folder(self, folder_name, parent_id=None):
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            file_metadata['parents'] = [parent_id]

        folder = self.service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

    def get_or_create_folder(self, folder_name, parent_id=None):
        # Check if the folder exists
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])

        if folders:
            # Folder exists
            return folders[0]['id']
        else:
            # Create the folder
            return self.create_folder(folder_name, parent_id)
    
    def upload_file(self, local_file_path, upload_name, folder_id=None):
        # File metadata
        file_metadata = {
            'name': upload_name,
            'mimeType': 'text/plain'
        }
        
        if folder_id:
            file_metadata['parents'] = [folder_id]

        media = MediaFileUpload(local_file_path, mimetype='text/plain')
        
        # Upload the file
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    
    def download_file(self, folder_id, filename, destination_path):
        query = f"'{folder_id}' in parents and name='{filename}'"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if files:
            file_id = files[0]['id']
        else:
            file_id = None
        
        if file_id:
            request = self.service.files().get_media(fileId=file_id)
        
            with open(destination_path, 'wb') as file:
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
    
