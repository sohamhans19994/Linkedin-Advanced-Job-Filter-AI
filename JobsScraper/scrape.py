import logging
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, \
    OnSiteOrRemoteFilters
import pandas as pd
import os
import re
import threading
import time
from datetime import datetime
from tqdm import tqdm
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from drive_utils import DriveUtils

class Scrape:
    def __init__(self,save_folder_path,config):
        self.jobs_save_path = os.path.join(save_folder_path,"jobs.csv")
        self.logs_save_path = os.path.join(save_folder_path,"logs.txt")
        logging.basicConfig(
            filename=self.logs_save_path,  # Log messages to this file
            level=logging.INFO,  # Capture all messages of DEBUG level and above
            filemode='a'  # Append to the log file (use 'w' to overwrite)
        )
        self.config = config
        self.drive_utils = DriveUtils(config)
        
    
    def linkedin_jobs(self):
        list_lock = threading.Lock()
        self.jobs_list = []
        self.just_saved = False
        def on_data(data: EventData):
            row = [data.title, data.company, data.company_link, data.date, data.link, data.insights,data.description,data.description_html,None]
            with list_lock:
                self.jobs_list.append(row)
        def on_error(error):
            logging.error(error)
        
        def on_metrics(metrics: EventMetrics):
            logging.info('[ON_METRICS]'+ str(metrics))

        def on_end():
            jobs_df = pd.DataFrame(self.jobs_list, columns=["Title", "Company", "Company Link", "Date", "Link", "Insights", "Description", "Description HTML","Match Score"])
            print(f"Found {len(self.jobs_list)} jobs")
            jobs_df.to_csv(self.jobs_save_path,index=False)
            logging.info('[END]')
        
        self.scraper = LinkedinScraper(
            chrome_executable_path='/usr/bin/chromedriver',
            chrome_binary_location=None, 
            chrome_options=None, 
            headless=True,  
            max_workers=10,  
            slow_mo=2,  
            page_load_timeout=40      
        )
        self.scraper.on(Events.DATA, on_data)
        self.scraper.on(Events.ERROR, on_error)
        self.scraper.on(Events.END, on_end)
        self.scraper.on(Events.METRICS,on_metrics)
    
        queries = [
            Query(
                query='Machine Learning OR AI OR Software Engineering',
                options=QueryOptions(
                    limit=10000,
                    locations=['United States'],
                    apply_link=False,  
                    skip_promoted_jobs=False,  
                    filters=QueryFilters(
                        relevance=RelevanceFilters.RECENT,
                        time=TimeFilters.DAY
                    )
                )
            )
        ]

        self.scraper.run(queries)

    def add_to_cloud(self):
        # folder_id = self.drive_utils.get_or_create_folder("Linkedin_scraped")
        folder_id = self.config["GDRIVE_FOLDER_ID"]
        cloud_save_name = os.path.basename(os.path.dirname(self.jobs_save_path)) + "_jobs.csv"
        self.drive_utils.upload_file(self.jobs_save_path,cloud_save_name,folder_id)
        print(f"Jobs csv added to cloud: {cloud_save_name}")

    def get_from_cloud(self, local_save_path):
        folder_id = self.config["GDRIVE_FOLDER_ID"]
        cloud_save_name = os.path.basename(os.path.dirname(self.jobs_save_path)) + "_jobs.csv"
        self.drive_utils.download_file(folder_id,cloud_save_name,local_save_path)