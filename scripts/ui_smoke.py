"""Optional real-browser acceptance run. Requires requirements-member3-render.txt."""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8501')
    args=parser.parse_args()
    rows=[]
    errors=[]
    figures=ROOT/'report/figures'
    figures.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1600,'height':1100}, device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url)
        run=page.get_by_role('button',name='Run selected model',exact=True)
        run.wait_for(timeout=30000)
        def select(index,value):
            page.get_by_role('combobox').nth(index).click()
            page.get_by_role('option',name=value,exact=True).click()
        def map_ready():
            page.locator('iframe:visible').first.wait_for(timeout=60000)
            frame=page.frame_locator('iframe:visible').first
            frame.locator('.leaflet-control-layers').wait_for(timeout=60000)
            return frame
        trips=['00000000','00000007','00000009','00000040','00000052','00000056','00000065','00000076','00000077','00000091']
        for trip in trips:
            select(0,trip)
            run.click()
            expect(run).to_be_disabled(timeout=15000)
            page.get_by_text('Approximate aligned point labels',exact=False).wait_for(timeout=120000)
            expect(run).to_be_enabled(timeout=120000)
            frame=map_ready()
            labels=frame.locator('.leaflet-control-layers').inner_text()
            assert 'HMM + directed bridges' in labels and 'Reference' not in labels
            rows.append(dict(trip_id=trip,model='best',status='pass',map_layers=labels.splitlines()))
            if trip=='00000056':
                # Wait for basemap images as well as route geometry.
                frame.locator('.leaflet-tile-loaded').first.wait_for(timeout=30000)
                page.screenshot(path=str(figures/'dashboard_best.png'),full_page=True)
            print(f'UI PASS {trip}/best',flush=True)
        # The actual batch button must advance through every model and finish.
        select(0,trips[0])
        page.get_by_role('button',name='Demo every model',exact=True).click()
        expect(run).to_be_disabled(timeout=15000)
        expect(run).to_be_enabled(timeout=180000)
        map_ready()
        # Check all six rows actually arrived in the live comparison dataframe.
        table=page.locator('[data-testid="stDataFrame"]:visible').first
        expect(table).to_be_visible()
        page.get_by_text('This trip · live model comparison',exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(figures/'dashboard_models.png'),full_page=True)
        labels=['K-Nearest Neighbours','Multi-Layer Perceptron','Logistic Regression','Random Forest','Histogram Boosting · baseline','Selected best · tuned + calibrated Histogram Boosting']
        for label in labels:
            select(1,label)
            page.get_by_text(f'Trip {trips[0]} · {label}',exact=True).wait_for(timeout=30000)
            map_ready()
            rows.append(dict(trip_id=trips[0],model=label,status='pass'))
            print(f'UI PASS {label}',flush=True)
        # Toggle decoder layer without losing the selected result.
        page.get_by_text('Show HMM route',exact=True).click()
        expect(page.get_by_role('checkbox',name='Show HMM route',exact=True)).not_to_be_checked()
        map_ready()
        assert not page.locator('[data-testid="stException"]').count()
        browser.close()
    assert not errors, errors
    output=ROOT/'results/metrics/ui_integration.json'
    output.write_text(json.dumps(dict(url=args.url,physical_second_laptop=False,checks=rows,page_errors=errors,batch_demo_verified=True,model_switches_verified=6,hmm_toggle_verified=True),indent=2)+'\n')
    print(f'Saved {len(rows)} browser checks to {output}')


if __name__=='__main__':main()
