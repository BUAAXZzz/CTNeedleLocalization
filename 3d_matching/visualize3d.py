"""Render reconstructed needles as PNG, standalone interactive HTML and VTK."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


HTML_VIEW = '''<!doctype html><meta charset="utf-8">
<title>CASE — 3D needle reconstruction</title>
<style>body{font:16px system-ui;margin:20px;background:#111827;color:#eee}a{color:#93c5fd}
canvas{width:100%;height:70vh;background:#050b16;touch-action:none;border:1px solid #374151}
label{margin-right:20px}button{padding:6px 15px}p{color:#cbd5e1}</style>
<h2>CASE: PRED reconstructed paths / PRIOR count prior</h2>
<p>Drag to rotate; scroll to zoom. Coordinates: patient LPS, millimeters.
Red = tip; green = handle; colored lines = predictions; gray dashed = complete reference paths.</p>
<label><input id="body" type="checkbox" checked>Body outline</label>
<label><input id="reference" type="checkbox" checked>Reference paths</label>
<button id="reset">Reset view</button><p>PRIOR_NOTE</p><canvas id="view"></canvas>
<script>
const data=DATA;
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
let yaw=-0.6,pitch=0.5,zoom=1,drag=null;
const all=[...data.body,...data.paths.flatMap(p=>[p.tip,p.handle]),...data.reference.flat()];
if(!all.length)all.push([0,0,0]);
const mins=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),
 maxs=[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))),
 center=mins.map((v,k)=>(v+maxs[k])/2),span=Math.max(1,...mins.map((v,k)=>maxs[k]-v));
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
 let a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;
 let c=Math.cos(pitch)*b-Math.sin(pitch)*z,d=Math.sin(pitch)*b+Math.cos(pitch)*z;
 const s=Math.min(canvas.width,canvas.height)*0.78*zoom/span;
 return [canvas.width/2+a*s,canvas.height/2-c*s,d];}
function line(a,b,color,width,dashed=false){a=project(a);b=project(b);
 ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[6,5]:[]);
 ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([]);}
function render(){const ratio=devicePixelRatio||1;
 canvas.width=Math.round(canvas.clientWidth*ratio);canvas.height=Math.round(canvas.clientHeight*ratio);
 ctx.clearRect(0,0,canvas.width,canvas.height);
 if(document.getElementById('body').checked){ctx.fillStyle='rgba(170,185,205,0.18)';
 for(const p of data.body){let a=project(p);ctx.fillRect(a[0],a[1],2*ratio,2*ratio);}}
 if(document.getElementById('reference').checked)
 for(const p of data.reference)line(p[0],p[1],'#a1a1aa',ratio,true);
 const paths=[...data.paths].sort((a,b)=>(project(a.tip)[2]+project(a.handle)[2])-(project(b.tip)[2]+project(b.handle)[2]));
 for(const p of paths){line(p.tip,p.handle,p.color,2*ratio);
 for(const [point,color] of [[p.tip,'#ff4040'],[p.handle,'#30e060']]){let a=project(point);
 ctx.fillStyle=color;ctx.beginPath();ctx.arc(a[0],a[1],3*ratio,0,7);ctx.fill();}
 let a=project(p.tip);ctx.fillStyle='#fff';ctx.font=`${11*ratio}px system-ui`;ctx.fillText(p.id,a[0]+5*ratio,a[1]-4*ratio);}
 ctx.fillStyle='#cbd5e1';ctx.font=`${12*ratio}px system-ui`;ctx.fillText('LPS coordinates (mm) • equal spatial scale',12*ratio,20*ratio);}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;
 drag=[e.clientX,e.clientY];render()};canvas.onpointerup=()=>drag=null;
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));render()};
for(const id of ['body','reference'])document.getElementById(id).onchange=render;
document.getElementById('reset').onclick=()=>{yaw=-.6;pitch=.5;zoom=1;render()};
window.onresize=render;render();
</script>'''


def physical(points, image):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return np.asarray(image.GetOrigin()) + (points*np.asarray(image.GetSpacing())) @ direction.T


def render_case(path, report):
    result = json.loads(path.read_text())
    case = path.parent.name
    image = sitk.ReadImage(result['ct_path'])
    volume = sitk.GetArrayFromImage(image)
    spacing = np.asarray(image.GetSpacing())
    tracks = [np.asarray([p['Tip'][:3], p['Handle'][:3]]) for p in result['pairs']]
    references = [np.asarray([n['centers_xyz']['2'], n['centers_xyz']['1']])
                  for n in report.get('needles', []) if not n['missing']]
    body_path = Path(result['ct_path']).parent/'body.nii.gz'
    cloud = np.empty((0, 3))
    if body_path.is_file():
        body_image = sitk.ReadImage(str(body_path))
        body = sitk.GetArrayFromImage(body_image) > 0
        # Slice-wise erosion extracts the torso outline without filling the
        # first and last acquired slices with artificial surface points.
        edge = body & ~ndimage.binary_erosion(body, structure=np.ones((1, 3, 3)))
        points = np.argwhere(edge)
        if len(points):
            points = points[::max(1, int(np.ceil(len(points)/1800)))][:, ::-1]
            cloud = physical(points, body_image)
    world_tracks = [physical(t, image) for t in tracks]
    world_references = [physical(t, image) for t in references]
    colors = [plt.cm.tab20(i % 20) for i in range(len(tracks))]
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(221, projection='3d')
    if len(cloud):
        ax.scatter(*cloud.T, s=1, c='gray', alpha=.07, depthshade=False)
    for t in world_references:
        ax.plot(*t.T, '--', color='gray', alpha=.45, linewidth=.7)
    for i, (t, color) in enumerate(zip(world_tracks, colors), 1):
        ax.plot(*t.T, color=color, linewidth=1.6)
        ax.scatter(*t[0], c='red', s=12)
        ax.scatter(*t[1], c='green', s=12)
        ax.text(*t[0], str(i), fontsize=6)
    bounds = [*world_tracks, *world_references]
    if len(cloud):
        bounds.append(cloud)
    if bounds:
        extent = np.ptp(np.concatenate(bounds), axis=0)
        ax.set_box_aspect(np.maximum(extent, 10))
    ax.set(xlabel='LPS x (mm)', ylabel='LPS y (mm)', zlabel='LPS z (mm)',
           title='3D reconstruction in patient coordinates')
    ax.view_init(elev=22, azim=-65)
    ax.legend(handles=[Line2D([0], [0], color='red', marker='o', linestyle='', label='tip'),
                       Line2D([0], [0], color='green', marker='o', linestyle='', label='handle'),
                       Line2D([0], [0], color='gray', linestyle='--', label='complete reference')], fontsize=7)
    # Projection axes are along the original CT grid, in millimeters. The
    # physical 3D view above additionally applies direction and origin.
    for position, collapse, dims, title in [(222, 0, (0, 1), 'Axial maximum projection'),
                                           (223, 1, (0, 2), 'Coronal maximum projection'),
                                           (224, 2, (1, 2), 'Sagittal maximum projection')]:
        ax = fig.add_subplot(position)
        mip = volume.max(axis=collapse)
        width, height = mip.shape[1], mip.shape[0]
        sx, sy = spacing[list(dims)]
        ax.imshow(mip, cmap='gray', vmin=-950, vmax=750,
                  extent=(-sx/2, (width-.5)*sx, (height-.5)*sy, -sy/2), aspect='equal')
        for t in references:
            p = t[:, dims]*spacing[list(dims)]
            ax.plot(p[:, 0], p[:, 1], '--', color='cyan', alpha=.45, linewidth=.6)
        for t, color in zip(tracks, colors):
            p = t[:, dims]*spacing[list(dims)]
            ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1)
            ax.scatter(p[0, 0], p[0, 1], c='red', s=10)
            ax.scatter(p[1, 0], p[1, 1], c='lime', s=10)
        labels = ['CT x (mm)', 'CT y (mm)', 'CT z (mm)']
        ax.set(title=title, xlabel=labels[dims[0]], ylabel=labels[dims[1]])
    prior = result['config']['n_prior']
    fig.suptitle('{} | reconstructed={} | count prior={} | complete reference={}'.format(
        case, len(tracks), prior, len(references)))
    fig.text(.5, .01, 'Original CT, no top-hat. Count prior source: {}. Reference paths are for visual comparison only.'.format(
        result['config'].get('prior_source', 'manual')), ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .96))
    fig.savefig(path.parent/'reconstruction.png', dpi=130)
    plt.close(fig)
    payload = {'body': np.round(cloud, 2).tolist(),
               'reference': [t.tolist() for t in world_references],
               'paths': [{'id': i, 'tip': t[0].tolist(), 'handle': t[1].tolist(),
                          'color': matplotlib.colors.to_hex(color)}
                         for i, (t, color) in enumerate(zip(world_tracks, colors), 1)]}
    page = HTML_VIEW.replace('DATA', json.dumps(payload, allow_nan=False))
    page = page.replace('CASE', html.escape(case)).replace('PRED', str(len(tracks)))
    page = page.replace('PRIOR_NOTE', 'Count prior source: '+html.escape(result['config'].get('prior_source', 'manual')))
    page = page.replace('PRIOR', str(prior))
    (path.parent/'interactive_3d.html').write_text(page)
    points = np.concatenate(world_tracks) if world_tracks else np.empty((0, 3))
    vtk = ['# vtk DataFile Version 3.0', 'Predicted needle trajectories, patient LPS millimeters',
           'ASCII', 'DATASET POLYDATA', 'POINTS {} float'.format(len(points))]
    vtk.extend('{} {} {}'.format(*p) for p in points)
    vtk.append('LINES {} {}'.format(len(tracks), 3*len(tracks)))
    vtk.extend('2 {} {}'.format(2*i, 2*i+1) for i in range(len(tracks)))
    if tracks:
        vtk.extend(['CELL_DATA {}'.format(len(tracks)), 'SCALARS score float 1', 'LOOKUP_TABLE default'])
        vtk.extend(str(p['score']) for p in result['pairs'])
    (path.parent/'needles_lps.vtk').write_text('\n'.join(vtk)+'\n')
    return {'case_id': case, 'n_prior': prior, 'predicted_paths': len(tracks),
            'complete_reference_paths': len(references), **result['counts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Optional reference endpoints for visual comparison')
    args = parser.parse_args()
    reports = {}
    if args.manifest:
        reports = {r['case_id']: r for r in json.loads(args.manifest.read_text())['cases']}
    rows, cards = [], []
    for path in sorted(args.root.glob('*/pred_3d.json')):
        row = render_case(path, reports.get(path.parent.name, {}))
        rows.append(row)
        case = html.escape(row['case_id'])
        cards.append('<section><h3><a href="{0}/interactive_3d.html">{0}: rotate 3D view</a></h3>'
                     '<p>Predicted {1} / prior {2}; complete reference {3}</p>'
                     '<a href="{0}/reconstruction.png"><img src="{0}/reconstruction.png"></a></section>'.format(
                         case, row['predicted_paths'], row['n_prior'], row['complete_reference_paths']))
        print('Rendered', row['case_id'], 'paths=', row['predicted_paths'], flush=True)
    if not rows:
        raise ValueError('No pred_3d.json files found')
    with (args.root/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'cases': len(rows), 'reconstructed_paths': sum(r['predicted_paths'] for r in rows),
               'count_prior_total': sum(r['n_prior'] for r in rows),
               'complete_reference_paths': sum(r['complete_reference_paths'] for r in rows),
               'empty_cases': [r['case_id'] for r in rows if not r['predicted_paths']], 'per_case': rows}
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>3D reconstruction</title>'
        '<style>body{font:16px system-ui;margin:25px;background:#eee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}'
        'section{background:white;padding:15px}img{width:100%}</style><h1>3D needle matching</h1>'
        '<p>Count priors in this run come from label instance counts, including incomplete needles. '
        'Reference paths are display-only; reconstructed paths use predicted endpoints. Counts are not accuracy metrics.</p>'
        '<main>'+''.join(cards)+'</main>')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_case'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
