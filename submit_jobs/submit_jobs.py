import datetime
import os
import pickle
import logging
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = "https://www.googleapis.com/auth/cloud-platform"

def get_creds():
	credentials = None
	# The file token.pickle stores the user's access and refresh tokens, and is
	# created automatically when the authorization flow completes for the first
	# time.
	if os.path.exists('token.pickle'):
		with open('token.pickle', 'rb') as token:
			credentials = pickle.load(token)
	# If there are no (valid) credentials available, let the user log in.
	if not credentials or not credentials.valid:
		if credentials and credentials.expired and credentials.refresh_token:
			credentials.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file('config/sa.json', SCOPES)
			credentials = flow.run_local_server(port=0)
		# Save the credentials for the next run
		with open('token.pickle', 'wb') as token:
			pickle.dump(credentials, token)

	return credentials

def submit_job(script_name,
			task_name, 
			image_url=None,
			scale_tier = None, 
			extra_args=None, 
			master_type=None, 
			accelerator_type=None):

	creds = get_creds()

	project_id = 'podcast-semantic-search'
	project_id = 'projects/{}'.format(project_id)
	
	job_name = task_name+"_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")

	args = ['--task', task_name]

	if extra_args:
		args += extra_args

	if not scale_tier:
		scale_tier = 'BASIC'

	training_inputs = {
		'args': args,
		'pythonVersion': '3.7',
		'scaleTier': scale_tier,
		'region': 'us-west1',
		'jobDir': "gs://ds-ml-nlp/ai_platform_job_files",
	}

	training_inputs.update({'packageUris': ["gs://ds-ml-nlp/ai_platform_job_files/lyrical_clustering-0.1.tar.gz"],
								'pythonModule': f'scripts.{script_name}',
								'runtimeVersion': '2.1',})

	if scale_tier == 'CUSTOM':
		training_inputs.update({'masterType': master_type})

	if accelerator_type:
		training_inputs.update({'masterConfig': {'acceleratorConfig': {'count': 1, 'type': accelerator_type}}})
		
	job_spec = {"jobId": job_name, "trainingInput": training_inputs}
	cloudml = discovery.build("ml", "v1", cache_discovery=False, credentials=creds)
	request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)
	try:
		response = request.execute()
	except HttpError as err:
		logging.error('There was an error creating the training job.'
					  ' Check the details:')
		logging.error(err._get_reason())

def gen_embeddings_Task(event, context,):
	return submit_job(task_name="gen_embeds",
						script_name="generate_embeddings",
						accelerator_type='NVIDIA-TESLA-K80', 
						master_type='n1-standard-4', 
						scale_tier='CUSTOM')

def scrape_lyrics_Task(event, context,):
	return submit_job(task_name="scrape_lyrics",
						script_name="scrape_genius",
						master_type='e2-standard-4', 
						scale_tier='CUSTOM')
