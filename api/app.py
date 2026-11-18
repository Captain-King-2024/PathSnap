"""FastAPI entry point for the PathSnap route viewer."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from api.server import DEFAULT_MODEL, Demo, ROOT, health

demo = Demo(Path(os.getenv('PATHSNAP_DATA', ROOT / 'data/processed')),
            Path(os.getenv('PATHSNAP_LABELS', ROOT / 'data/labels')),
            Path(os.getenv('PATHSNAP_MODEL', DEFAULT_MODEL)),
            os.getenv('PATHSNAP_SPLIT', 'student_test'))
app = FastAPI(title='PathSnap')


@app.get('/')
def index():
    return FileResponse(ROOT / 'ui/index.html', media_type='text/html')


@app.get('/api/trips')
def trips():
    return {'trips': demo.trip_ids, 'split': demo.split}


@app.get('/api/health')
def health_check():
    return health(demo)


@app.get('/api/match/{trip}')
def match(trip: str):
    try:
        return demo.match(trip)
    except KeyError as error:
        raise HTTPException(404, 'Trip not found') from error
