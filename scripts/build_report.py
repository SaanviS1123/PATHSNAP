"""Assemble measured member sources into a single, reproducible PDF."""
from pathlib import Path
import copy
import hashlib
import html
import json
import re
import textwrap
import xml.etree.ElementTree as ET
import markdown
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether, Preformatted
import os
os.environ.setdefault("MPLCONFIGDIR",str(ROOT_CACHE := Path(__file__).resolve().parents[1]/".cache/matplotlib"))
import matplotlib

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report'
SOURCES=[REPORT/'member1/eda_and_models.md',REPORT/'member2/models_and_tuning.md',REPORT/'member3/hmm_and_maps.md']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble():
    integration=pd.read_csv(ROOT/'results/metrics/integration_run.csv',dtype={'trip_id':str})
    best=integration[integration.model=='best']
    browser_path=ROOT/'results/metrics/ui_integration.json'
    browser=json.loads(browser_path.read_text()) if browser_path.exists() else None
    a,b,c=[p.read_text() for p in SOURCES]
    a=a[a.index('## 2.2'):]
    eda, member1_models=a.split('## 3. Member 1 model experiments',1)
    member1_models=member1_models[member1_models.index('### 3.2 KNN'):]
    member1_models=member1_models.split('## Reproducibility, handoff and scope')[0].replace('### 3.2 ','### ')
    b=b[b.index('## 3.1'):]
    models,tuning=b.split('## 4.',1)
    c=c[c.index('## 5.'):]
    c=c.split('## Reproducibility and handoff')[0]
    overview='''# PATHSNAP

Distributed map matching and route reconstruction

## 1. Objective and delivered system

Given noisy GPS observations and a directed road network, choose a road arc for every observation and reconstruct a feasible route. The system compares five classifier families, selects and calibrates a model using training-trip validation, then decodes a weighted HMM-style sequence with Viterbi. A browser dashboard demonstrates every family and the final tuned/calibrated configuration.

All results use the organiser's train and student-test partitions. Evaluator-test data was not accessed. Point labels are approximate ordered-route alignments, not supplied point-level ground truth. Student test is a development holdout, not proof of deployment accuracy. Family selection, tuning and candidate configuration share training evidence; CV estimates are not nested or fully independent.

Code, analysis and report drafting used Codex AI assistance. The team remains responsible for reviewing, understanding and reproducing the work. No member names or physical second-laptop evidence are invented.

## 2.1 Data and method overview

Each trip supplies headerless track (longitude, latitude, timestamp), nodes (longitude, latitude), directed arcs (from-node, to-node), and a shorter ordered reference route of local arc IDs. Loaders check finite coordinates and identifier ranges. Inference reads track, nodes and arcs only. Offline training aligns observations to ordered route occurrences; server-side evaluation opens the reference only after inference.

Pipeline: local metric projection -> indexed candidates -> twelve schema features -> positive candidate probabilities -> per-point classifier choices / directed HMM -> separate reference evaluation -> API -> dashboard. No saved prediction file substitutes for a live request.

'''
    section7=f'''## 7. API, dashboard and deployment

### Two-laptop request flow

The model laptop hosts FastAPI on loopback port 8000 and Streamlit on port 8501. Cloudflared exposes only the dashboard. The second laptop opens its temporary HTTPS dashboard link. Streamlit executes API requests on the host, so raw files, models and reference routes remain there. This follows the latest requested deployment; it supersedes the original plan to run a UI server on Laptop 2.

`GET /health`, `/models` and `/trips` expose readiness, model selection evidence and permitted trip IDs. `POST /predict` is a synchronous compatibility interface. The UI uses `POST /jobs`, polls `/jobs/{{id}}`, then fetches `/jobs/{{id}}/result`. Only eight-digit permitted trip IDs and explicit model names are accepted; extra request fields are rejected. Public response schemas contain no reference IDs, aligned labels or reference geometry. No browser-to-API CORS exception is needed.

The final calibrated scorer and frozen HMM load at startup; other scorer artifacts load once on demand with schema checks. One worker serializes inference and bounds memory/CPU use. At most four jobs are outstanding and twelve jobs/results are retained. Jobs expire on eviction or server restart. Polling avoids a long in-flight tunnel request during KNN. The scorer owns its preprocessing and calibration, applied once.

`api/evaluation.py` is a separate post-inference boundary. It loads the reference, regenerates approximate alignment labels, and calls the Member 3 pointwise, sequence and direct-continuity metrics. Missing references leave valid predictions available with metrics marked unavailable. Log lines show trip ID, model, point count, server latency and classifier/HMM accuracy. Full errors stay in the host log.

### Model demonstration and interpretation

The dashboard offers KNN, MLP, Logistic Regression, Random Forest, baseline Histogram Boosting and the final tuned/calibrated Histogram Boosting. "Demo every model" runs them in that order and finishes with the selected configuration. Results accumulate in a per-trip table; the user can revisit each map, toggle HMM, inspect stage timings, and download prediction JSON. GPS observations, classifier arcs and HMM roads are distinct map layers. Each road arc is drawn separately so disconnected states never create fictitious straight road connectors. There is no live reference layer.

The best badge is based on highest mean training-fold observation Top-1. It is not recalculated from one displayed trip. The frozen decoder was tuned for the selected calibrated model; applying it to other classifiers is an interactive demonstration, not a separately optimized HMM comparison. Historical comparison rows are classifier-only; live per-trip metrics do not replace the full holdout study.

### Executed integration evidence

`results/metrics/integration_run.csv` records {len(integration)} successful real HTTP job results: all {len(best)} student trips with the selected model and the five other configurations on the first trip. Every returned observation has a valid predicted arc. Selected-model server time (including fresh reference alignment and response construction, excluding queue wait) ranged from {best.server_ms.min():.2f} to {best.server_ms.max():.2f} ms. HTTP wall time includes polling and JSON transfer and is recorded separately. The first-trip KNN request took {integration.loc[integration.model=='knn','server_ms'].iloc[0]/1000:.2f} s. This is measured API latency, not a claim about other hardware or network conditions.

'''
    columns=['trip_id','model','n_observations','server_ms','classifier_accuracy','hmm_accuracy']
    section7+='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for row in integration[columns].itertuples(index=False,name=None):
        section7+='| '+' | '.join(f'{value:.4f}' if isinstance(value,float) else str(value) for value in row)+' |\n'
    section7+='\n'
    if browser:
        section7+=f"`ui_integration.json` records {len(browser['checks'])} successful real-browser checks, covering ten student trips, the batch model demo, switching among six completed results and the HMM toggle. Browser JavaScript errors: {len(browser['page_errors'])}. The following screenshots show the actual local dashboard; they are not evidence of a physical second laptop.\n\n"
        section7+='![Live dashboard: largest student trip, selected model](figures/dashboard_best.png)\n\n![Live dashboard: all model configurations on one trip](figures/dashboard_models.png)\n\n'
    else:
        section7+='Browser acceptance evidence is not yet available.\n\n'
    section7+='''### Commands, connectivity and remaining external check

All commands are in `scripts/README.md`. `scripts/setup.sh` installs exact direct dependency versions for Python 3.13. `scripts/host.sh` starts API, UI and the temporary dashboard tunnel and stops its children on Ctrl-C. Separate-terminal commands are `serve.sh api`, `serve.sh ui`, and `tunnel.sh`. The receiving laptop may run `check-dashboard.sh HTTPS_URL` or curl `HTTPS_URL/_stcore/health`, then open the URL and run a prediction while the host log is visible.

Public tunnel verification remains blocked in this development environment: cloudflared 2026.11.1 and a direct HTTPS check both received connection resets from `api.trycloudflare.com` before a tunnel URL could be created. No working public URL, remote terminal screenshot or physical second-laptop success is claimed. Repeat the connectivity check on the actual demo network and capture the receiving browser plus host log. Quick Tunnels are temporary testing links and change URL on restart.

Fallback: place both laptops on the same Wi-Fi/hotspot, run `UI_HOST=0.0.0.0 scripts/host.sh --local`, allow inbound dashboard port 8501, and open `http://HOST_LAN_IP:8501` on Laptop 2. API remains local. If QUIC connectivity fails after tunnel creation, `TUNNEL_PROTOCOL=http2 scripts/tunnel.sh` is available. If Internet tiles fail, prediction geometry/metrics still work over LAN; do not present saved screenshots as live inference.

## 8. Findings and limitations

The selected calibrated classifier and directed HMM improve aggregate student-test point accuracy and route quality, while retaining a visible failure case. Candidate misses, imperfect label alignment, bounded route searches, long gaps and model probability quality remain limits. Model selection is a training-CV decision; the very small baseline CV lead and slight tuned student Top-1 decline are disclosed in section 4. This demonstration does not establish performance on unseen regions, evaluator-test data, production traffic or a network not tested here.

The shipped scripts reproduce model/API/UI checks and assemble this report. Source tables, plots and experiment logs remain in the repository; raw and regenerable processed data remain local. `results/metrics/report_manifest.json` records SHA-256 values for report inputs and evidence. A physical two-laptop run is the remaining deployment acceptance step.

## 9. References and evidence index

- Newson and Krumm (2009), Hidden Markov Map Matching Through Noise and Sparseness. https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/
- Cloudflare Quick Tunnels: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
- Streamlit deployment/configuration: https://docs.streamlit.io/develop/api-reference/configuration/config.toml
- Data and feature evidence: `inventory.csv`, `quality_report.csv`, `label_alignment.csv`, `candidate_sweep.csv`, `data_manifest.csv`.
- Classifier evidence: `model_comparison.csv`, `tuned_vs_untuned.csv`, `tuning_trials.csv`, `calibration_audit.json`, `final_export_metrics.csv`, model cards and experiment logs.
- Decoder evidence: `hmm_tuning.csv`, `hmm_comparison.csv`, per-trip comparison, `decode_timing.csv`, `folium_render_checks.json` and frozen `configs/hmm.json`.
- Deployment evidence: `integration_run.csv`, `ui_integration.json`, `report/figures/`, concise `DECISIONS.md`, and reproducible commands in `scripts/README.md`.
'''
    text=overview+eda+'\n'+models+'\n'+member1_models+'\n## 4.'+tuning+'\n'+c+'\n'+section7
    text=text.replace('(../../results/','(../results/')
    target=REPORT/'PATHSNAP_report.md'
    target.write_text(text)
    return target


def build_pdf(source):
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),('BodyItalic','DejaVuSans-Oblique.ttf'),('Code','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Text',fontName='Body',fontSize=9,leading=13,spaceAfter=7,splitLongWords=True))
    styles.add(ParagraphStyle(name='Cell',parent=styles['Text'],fontSize=6.6,leading=8.7,spaceAfter=0))
    styles.add(ParagraphStyle(name='Caption',parent=styles['Text'],fontSize=8,leading=11,textColor=colors.HexColor('#55665f')))
    for key,size,leading in [('Heading1',26,31),('Heading2',16,21),('Heading3',12,17),('Heading4',10,14)]:
        styles[key].fontName='BodyBold'; styles[key].fontSize=size; styles[key].leading=leading
        styles[key].spaceBefore=14;styles[key].spaceAfter=10;styles[key].textColor=colors.HexColor('#164c3c')
    width=A4[0]-88
    story=[]
    root=ET.fromstring('<root>'+markdown.markdown(source.read_text(),extensions=['tables','fenced_code'])+'</root>')

    def inline(element):
        cloned=copy.deepcopy(element)
        def convert(node):
            for child in list(node):convert(child)
            if node.tag=='code':node.tag='font';node.attrib={'name':'Code'}
            elif node.tag=='strong':node.tag='b'
            elif node.tag=='em':node.tag='i'
            elif node.tag=='a':
                href=node.get('href','')
                if href.startswith('https://'):node.tag='link';node.attrib={'href':href,'color':'#176b70'}
                else:node.tag='span';node.attrib={}
        convert(cloned)
        return (html.escape(cloned.text or '')+''.join(ET.tostring(child,encoding='unicode') for child in cloned))

    def image_node(node):
        path=(source.parent/node.get('src')).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        img=Image(str(path));ratio=min(width/img.imageWidth,350/img.imageHeight)
        img.drawWidth=img.imageWidth*ratio;img.drawHeight=img.imageHeight*ratio
        story.append(KeepTogether([Spacer(1,8),img,Spacer(1,5),Paragraph(html.escape(node.get('alt','')),styles['Caption']),Spacer(1,9)]))

    def block(node):
        if node.tag in ('h1','h2','h3','h4'):
            title=''.join(node.itertext())
            if node.tag=='h2' and re.match(r'^[3-9]\.\s',title):story.append(PageBreak())
            story.append(Paragraph(inline(node),styles['Heading'+node.tag[1]]))
        elif node.tag=='p':
            if node.find('img') is not None:
                for image in node.findall('img'):image_node(image)
            else:story.append(Paragraph(inline(node),styles['Text']))
        elif node.tag in ('ul','ol'):
            for index,item in enumerate(node.findall('li')):
                prefix=f'{index+1}. ' if node.tag=='ol' else '- '
                story.append(Paragraph(prefix+inline(item),styles['Text']))
        elif node.tag=='pre':
            lines=['\n'.join(textwrap.wrap(line,88,replace_whitespace=False,drop_whitespace=False)) for line in ''.join(node.itertext()).splitlines()]
            story.append(Preformatted('\n'.join(lines),ParagraphStyle(name='Pre',fontName='Code',fontSize=7,leading=10,spaceAfter=10)))
        elif node.tag=='table':
            rows=node.findall('.//tr')
            data=[]
            for row_index,row in enumerate(rows):
                cells=[]
                for cell in row:
                    if row_index==0:
                        label=html.escape(''.join(cell.itertext()).replace('_',' '))
                    else:label=inline(cell)
                    cells.append(Paragraph(label,styles['Cell']))
                data.append(cells)
            count=len(data[0])
            # Content-weighted widths with enough room for dense measurement tables.
            weights=[]
            for column in range(count):
                weights.append(min(24,max(8,max(len(''.join(row[column].itertext())) for row in rows))))
            widths=[width*w/sum(weights) for w in weights]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebe3')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#ced8d2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f5')]),
                ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,12)])
        elif node.tag=='hr':story.append(Spacer(1,12))
        else:
            for child in node:block(child)
    for node in root:block(node)
    output=REPORT/'PATHSNAP_report.pdf'
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFont('Body',7);canvas.setFillColor(colors.HexColor('#55665f'))
        canvas.drawString(44,24,'PATHSNAP | Measured development results | 18 November 2026')
        canvas.drawRightString(A4[0]-44,24,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=42,
                      title='PATHSNAP - Map Matching and Route Reconstruction',author='PATHSNAP team').build(story,onFirstPage=decorate,onLaterPages=decorate)
    inputs=SOURCES+[source,Path(__file__).resolve(),ROOT/'requirements.txt',ROOT/'configs/hmm.json']
    inputs+=sorted((ROOT/'results/metrics').glob('*.csv'))
    inputs+=[p for p in [ROOT/'results/metrics/ui_integration.json',ROOT/'results/metrics/calibration_audit.json'] if p.exists()]
    inputs+=sorted((REPORT/'figures').glob('*.png'))
    manifest=dict(report=str(output.relative_to(ROOT)),report_sha256=sha(output),
                  inputs={str(path.relative_to(ROOT)):sha(path) for path in inputs},
                  physical_second_laptop_verified=False,public_tunnel_verified=False)
    (ROOT/'results/metrics/report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output)


if __name__=='__main__':build_pdf(assemble())
