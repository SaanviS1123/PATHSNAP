"""Render/verify actual Folium pages in an isolated headless Chromium context."""
import argparse
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from src.models.training import ROOT,write_json
from src.data_processing.loaders import load_trip


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser-cache',type=Path)
    args=parser.parse_args()
    if args.browser_cache:
        os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(args.browser_cache.resolve())
    from playwright.sync_api import sync_playwright
    timing=pd.read_csv(ROOT/'results/metrics/decode_timing.csv',dtype={'trip_id':str})
    largest=timing[timing.partition=='student_test'].sort_values('n_obs').iloc[-1].trip_id
    changes=pd.read_csv(ROOT/'results/metrics/hmm_pointwise_changes.csv',dtype={'trip_id':str})
    worst=changes.sort_values('hmm_change').iloc[0].trip_id
    evidence=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1280,'height':850},device_scale_factor=1)
        for path in sorted((ROOT/'maps').glob('pred_*.html')):
            page=context.new_page()
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            text=path.read_text()
            map_name=re.search(r'var\s+(map_[a-f0-9]+)\s*=\s*L\.map',text).group(1)
            page.goto(path.as_uri(),wait_until='load',timeout=60000)
            page.wait_for_function('(name) => window[name] && window[name]._loaded',arg=map_name,timeout=30000)
            assert page.locator('.leaflet-control-layers').is_visible()
            assert page.locator('.leaflet-container').is_visible()
            if errors:
                raise RuntimeError(f'{path.name}: JavaScript errors: {errors}')
            trip_id=path.stem.split('_')[-1]
            screenshot=None
            observation=None
            focus_reason=None
            if trip_id in {largest,worst}:
                prediction=pd.read_parquet(ROOT/f'data/processed/member3/prediction_{trip_id}.parquet')
                differing=prediction[prediction.classifier_arc!=prediction.hmm_arc].obs_idx.to_numpy()
                focus_reason='median classifier/HMM disagreement'
                if trip_id==worst:
                    labels=pd.read_parquet(ROOT/'data/processed/obs_labels.parquet')
                    joined=prediction.merge(labels[labels.trip_id==trip_id][['obs_idx','ref_arc']],on='obs_idx')
                    failures=joined[(joined.ref_arc>=0)&(joined.classifier_arc==joined.ref_arc)&(joined.hmm_arc!=joined.ref_arc)].obs_idx.to_numpy()
                    if len(failures):
                        differing=failures
                        focus_reason='median known-label classifier-correct/HMM-wrong case; offline illustration only'
                observation=int(differing[len(differing)//2]) if len(differing) else len(prediction)//2
                trip=load_trip(ROOT/'data/raw/student_test'/trip_id)
                lon,lat=trip.track[observation,:2]
                # Zoom comes from a real disagreement point, not a fabricated map.
                page.evaluate('(data) => { window[data.name].stop(); window[data.name].setView(data.location,16,{animate:false}); }',dict(name=map_name,location=[float(lat),float(lon)]))
                page.get_by_text('Road network (cropped)',exact=True).click()
                page.wait_for_function("() => Array.from(document.querySelectorAll('.leaflet-tile')).every(image => image.complete)",timeout=30000)
                page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                screenshot='hmm_folium_screenshot.png' if trip_id==largest else 'hmm_failure_screenshot.png'
                page.screenshot(path=str(ROOT/'results/plots'/screenshot),full_page=True)
            tiles=page.locator('.leaflet-tile').evaluate_all('(images) => ({total:images.length,loaded:images.filter(image=>image.naturalWidth>0).length})')
            evidence.append(dict(trip_id=trip_id,map=path.name,javascript_errors=errors,layer_control=True,
                                 tiles=tiles,screenshot=screenshot,focus_observation=observation,focus_reason=focus_reason,map_zoom=page.evaluate('(name) => window[name].getZoom()',map_name),browser_version=browser.version))
            page.close()
        context.close();browser.close()
    write_json(ROOT/'results/metrics/folium_render_checks.json',evidence)
    print(f'Verified {len(evidence)} Folium pages; captured largest-trip and lowest-gain views')


if __name__=='__main__':
    main()
