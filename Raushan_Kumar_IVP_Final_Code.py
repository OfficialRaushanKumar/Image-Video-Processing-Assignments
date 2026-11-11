#!/usr/bin/env python3
"""
Image and Video Processing Assignment
Student: Raushan Kumar | Reg. No.: RA2311031010126
Subject: 21CSE480T - Image and Video Processing

Implements all 10 assignment problems using Python, OpenCV and NumPy.
If no input video is supplied, a clearly-labelled synthetic road-scene video
is generated for demonstration. Replace it with your own permitted input
video for the final practical submission.
"""
from __future__ import annotations
import argparse, json, csv, math, os, shutil, subprocess
from pathlib import Path
import cv2
import numpy as np

STUDENT = "Raushan Kumar"
REG_NO = "RA2311031010126"

def ensure_dirs(root: Path):
    for name in ("output_images", "extracted_frames", "processed_videos", "metrics"):
        (root / name).mkdir(parents=True, exist_ok=True)

def make_synthetic_video(path: Path, width=640, height=360, fps=10, frames=100):
    """Create a synthetic road-scene video for reproducible demonstration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create demo video. Check OpenCV video codec support.")
    for i in range(frames):
        f = np.zeros((height, width, 3), np.uint8)
        # Sky and road
        f[:] = (185, 150, 105)
        horizon = int(height * .48)
        cv2.rectangle(f, (0, horizon), (width, height), (65, 65, 65), -1)
        # Road edges and lane markings
        cv2.line(f, (int(width*.08), height), (int(width*.42), horizon), (235,235,235), 4)
        cv2.line(f, (int(width*.92), height), (int(width*.58), horizon), (235,235,235), 4)
        for y in range(horizon+15, height, 55):
            x = width//2 + int((y-horizon)*.06)
            cv2.line(f, (x-3,y), (x+3,y+25), (245,245,245), 4)
        # roadside buildings/trees
        cv2.rectangle(f, (10,horizon-65), (105,horizon), (95,110,125), -1)
        cv2.rectangle(f, (width-100,horizon-85), (width-15,horizon), (105,95,85), -1)
        for x in (135, 520):
            cv2.circle(f, (x,horizon-20), 22, (35,105,35), -1)
            cv2.rectangle(f,(x-5,horizon-20),(x+5,horizon+5),(40,45,35),-1)
        # Moving vehicle
        cx = int(width*.25 + (i*4) % int(width*.5))
        cy = int(height*.68 + 12*math.sin(i/8))
        cv2.rectangle(f, (cx-42,cy-22), (cx+42,cy+18), (35,45,185), -1)
        cv2.rectangle(f, (cx-25,cy-38), (cx+25,cy-18), (45,95,150), -1)
        cv2.circle(f,(cx-27,cy+18),9,(15,15,15),-1); cv2.circle(f,(cx+27,cy+18),9,(15,15,15),-1)
        cv2.putText(f, "SYNTHETIC ROAD SCENE", (12,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),2)
        writer.write(f)
    writer.release()
    return {"width":width,"height":height,"fps":fps,"frames":frames,"duration_s":frames/fps,
            "source":"synthetic demonstration video (not a real traffic recording)"}

def read_video_info(path: Path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open video: {path}")
    info={"frames":int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
          "fps":float(cap.get(cv2.CAP_PROP_FPS)),
          "width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    info["duration_s"]=info["frames"]/info["fps"] if info["fps"] else 0
    info["size_kb"]=round(path.stat().st_size/1024,2)
    return info

def mse_psnr(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.shape != b.shape: b=cv2.resize(b,(a.shape[1],a.shape[0]))
    m=float(np.mean((a-b)**2))
    p=float("inf") if m==0 else 10*math.log10((255.0**2)/m)
    return round(m,4), ("lossless" if math.isinf(p) else round(p,2))

def save_json(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=str),encoding="utf-8")

def read_first_frame(video):
    cap=cv2.VideoCapture(str(video)); ok,frame=cap.read(); cap.release()
    if not ok: raise RuntimeError("Could not read first video frame.")
    return frame

def p1_edges(frame, out):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    sx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); sy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag=cv2.convertScaleAbs(cv2.magnitude(sx,sy))
    can=cv2.Canny(gray,100,200)
    cv2.imwrite(str(out/"p1_original_gray.png"),gray)
    cv2.imwrite(str(out/"p1_sobel.png"),mag)
    cv2.imwrite(str(out/"p1_canny.png"),can)
    return {"sobel_output":"p1_sobel.png","canny_output":"p1_canny.png",
            "observation":"Sobel highlights intensity gradients; Canny uses smoothing, non-maximum suppression and hysteresis. Fine texture may remain."}

def p2_compression(frame,out):
    original_png=out/"p2_original.png"; cv2.imwrite(str(original_png),frame)
    raw_bytes=frame.nbytes
    results=[]
    # PNG lossless round-trip
    ok,png=cv2.imencode(".png",frame)
    dec_png=cv2.imdecode(png,cv2.IMREAD_COLOR)
    mse,psnr=mse_psnr(frame,dec_png)
    results.append({"method":"PNG lossless","encoded_size_kb":round(len(png)/1024,2),
                    "compression_ratio_raw_vs_encoded":round(raw_bytes/len(png),3),"mse":mse,"psnr_db":psnr})
    for q in (90,50,20):
        ok,enc=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,q])
        dec=cv2.imdecode(enc,cv2.IMREAD_COLOR)
        m,p=mse_psnr(frame,dec)
        results.append({"method":f"JPEG quality {q}","encoded_size_kb":round(len(enc)/1024,2),
                        "compression_ratio_raw_vs_encoded":round(raw_bytes/len(enc),3),"mse":m,"psnr_db":p})
        cv2.imwrite(str(out/f"p2_jpeg_q{q}.jpg"),dec)
    cv2.imwrite(str(out/"p2_png_lossless.png"),dec_png)
    return results

def p3_segmentation(frame,out):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    t,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
    # Seeded region growing: 8-connected flood fill within intensity tolerance.
    seed=(frame.shape[1]//2, int(frame.shape[0]*.72))
    lo=18; hi=18
    mask=np.zeros((gray.shape[0]+2,gray.shape[1]+2),np.uint8)
    grown=gray.copy()
    cv2.floodFill(grown,mask,seed,120,loDiff=lo,upDiff=hi,flags=8|cv2.FLOODFILL_MASK_ONLY|((255)<<8))
    region=mask[1:-1,1:-1]*255
    # Watershed with simple foreground/background markers
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,sure_fg=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel=np.ones((3,3),np.uint8)
    sure_bg=cv2.dilate(sure_fg,kernel,iterations=2)
    dist=cv2.distanceTransform(sure_fg,cv2.DIST_L2,5)
    _,fg=cv2.threshold(dist,.35*dist.max(),255,0)
    fg=np.uint8(fg); unknown=cv2.subtract(sure_bg,fg)
    n,markers=cv2.connectedComponents(fg); markers=markers+1; markers[unknown==255]=0
    ws=frame.copy(); cv2.watershed(ws,markers); ws[markers==-1]=(0,0,255)
    for name,img in (("otsu",otsu),("adaptive",adaptive),("region_growing",region),("watershed",ws)):
        cv2.imwrite(str(out/f"p3_{name}.png"),img)
    return {"otsu_threshold":float(t),"region_seed_xy":seed,"region_pixels":int(np.count_nonzero(region)),
            "watershed_boundary_pixels":int(np.count_nonzero(markers==-1))}

def p4_extract(video,root,stride=20):
    cap=cv2.VideoCapture(str(video)); i=0; saved=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%stride==0:
            name=f"frame_{i:04d}.png"; cv2.imwrite(str(root/"extracted_frames"/name),frame); saved.append(i)
        i+=1
    cap.release()
    return {"stride":stride,"frames_read":i,"frames_saved":len(saved),"zero_based_indices":saved}

def p5_sampling(video,root):
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
    frames=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    results=[]
    for factor,label in ((1,"original"),(2,"half_rate"),(4,"quarter_rate")):
        out=root/"processed_videos"/f"p5_{label}.mp4"
        writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),max(fps/factor,1),(w,h))
        chosen=frames[::factor]
        for f in chosen: writer.write(f)
        writer.release()
        results.append({"version":label,"fps":round(fps/factor,2),"frames":len(chosen),"size_kb":round(out.stat().st_size/1024,2)})
    return results

def p6_transform(frame,out):
    h,w=frame.shape[:2]; center=(w/2,h/2)
    M=np.float32([[1,0,40],[0,1,25]])
    translated=cv2.warpAffine(frame,M,(w,h),borderMode=cv2.BORDER_CONSTANT)
    R=cv2.getRotationMatrix2D(center,30,1.0); rotated=cv2.warpAffine(frame,R,(w,h))
    scaled=cv2.resize(frame,None,fx=.6,fy=.6,interpolation=cv2.INTER_AREA)
    canvas=np.zeros_like(frame); sh,sw=scaled.shape[:2]; y=(h-sh)//2; x=(w-sw)//2; canvas[y:y+sh,x:x+sw]=scaled
    for name,img in (("translation",translated),("rotation",rotated),("scaling",canvas)):
        cv2.imwrite(str(out/f"p6_{name}.png"),img)
    return {"translation_pixels":[40,25],"rotation_degrees":30,"scale_factor":0.6,
            "resized_content_dimensions":[sw,sh],"video_canvas_dimensions":[w,h]}

def p7_difference(video,out):
    cap=cv2.VideoCapture(str(video)); frames=[]
    for _ in range(31):
        ok,f=cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    if len(frames)<2: raise RuntimeError("Need at least two frames for frame differencing.")
    a,b=frames[0],frames[-1]
    diff=cv2.absdiff(a,b); gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    _,mask=cv2.threshold(gray,30,255,cv2.THRESH_BINARY)
    cv2.imwrite(str(out/"p7_frame_a.png"),a); cv2.imwrite(str(out/"p7_frame_b.png"),b)
    cv2.imwrite(str(out/"p7_absolute_difference.png"),diff); cv2.imwrite(str(out/"p7_change_mask.png"),mask)
    return {"compared_frame_indices":[0,len(frames)-1],"mean_absolute_difference":round(float(gray.mean()),3),
            "changed_pixels":int(np.count_nonzero(mask)),"changed_pixel_percent":round(100*np.count_nonzero(mask)/mask.size,3)}

def p8_filter_video(video,root):
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
    paths=[root/"processed_videos"/f"p8_{n}.mp4" for n in ("median5x5","gaussian5x5","bilateral")]
    writers=[cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h)) for p in paths]
    count=0; psnr_acc=[[],[],[]]
    rng=np.random.default_rng(42)
    while True:
        ok,frame=cap.read()
        if not ok: break
        noisy=frame.copy()
        # Add light Gaussian noise for a repeatable filtering comparison.
        noise=rng.normal(0,8,frame.shape).astype(np.int16)
        noisy=np.clip(frame.astype(np.int16)+noise,0,255).astype(np.uint8)
        filtered=[cv2.medianBlur(noisy,5),cv2.GaussianBlur(noisy,(5,5),0),cv2.bilateralFilter(noisy,7,50,50)]
        for j,(wr,flt) in enumerate(zip(writers,filtered)):
            wr.write(flt); psnr_acc[j].append(mse_psnr(frame,flt)[1])
        if count==0:
            cv2.imwrite(str(root/"output_images"/"p8_noisy_frame.png"),noisy)
            for nm,flt in zip(("median5x5","gaussian5x5","bilateral"),filtered):
                cv2.imwrite(str(root/"output_images"/f"p8_{nm}.png"),flt)
        count+=1
    cap.release()
    for wr in writers: wr.release()
    return {"frames_processed":count,"median_kernel":"5x5","psnr_db_mean":[round(float(np.mean(x)),2) for x in psnr_acc],
            "outputs":[p.name for p in paths]}

def ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def p9_compress(video,root):
    results=[]
    if ffmpeg_available():
        for crf,label in ((18,"high_quality"),(28,"balanced"),(38,"low_bandwidth")):
            out=root/"processed_videos"/f"p9_{label}_crf{crf}.mp4"
            cmd=["ffmpeg","-y","-i",str(video),"-c:v","libx264","-preset","veryfast","-crf",str(crf),"-an",str(out)]
            run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if run.returncode==0 and out.exists():
                size=out.stat().st_size
                results.append({"preset":label,"crf":crf,"file_size_kb":round(size/1024,2),"status":"encoded with FFmpeg H.264"})
            else: results.append({"preset":label,"crf":crf,"status":"FFmpeg encoding failed"})
    else:
        # OpenCV fallback is explicitly identified; it does not emulate CRF.
        for label in ("high_quality","balanced","low_bandwidth"):
            out=root/"processed_videos"/f"p9_{label}_opencv_fallback.mp4"
            cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
            wr=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
            while True:
                ok,f=cap.read()
                if not ok: break
                wr.write(f)
            cap.release(); wr.release()
            results.append({"preset":label,"file_size_kb":round(out.stat().st_size/1024,2),"status":"OpenCV MP4V fallback; CRF not applied"})
    return results

def p10_bitrate(video,root):
    results=[]
    if ffmpeg_available():
        for kbps in (200,500,1000):
            out=root/"processed_videos"/f"p10_stream_{kbps}kbps.mp4"
            cmd=["ffmpeg","-y","-i",str(video),"-c:v","libx264","-b:v",f"{kbps}k","-maxrate",f"{kbps}k","-bufsize",f"{2*kbps}k","-an",str(out)]
            run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if run.returncode==0 and out.exists():
                duration=max(read_video_info(out)["duration_s"],.001)
                actual=out.stat().st_size*8/duration/1000
                results.append({"target_kbps":kbps,"actual_kbps":round(actual,2),"size_kb":round(out.stat().st_size/1024,2),"status":"FFmpeg H.264"})
            else: results.append({"target_kbps":kbps,"status":"FFmpeg encoding failed"})
    else:
        results.append({"status":"FFmpeg not installed; bitrate ladder not generated. Install FFmpeg and rerun."})
    return results

def main():
    parser=argparse.ArgumentParser(description="Run all 10 Image and Video Processing assignment problems.")
    parser.add_argument("video",nargs="?",help="Input video path (optional; synthetic demo is generated if omitted).")
    parser.add_argument("--output",default="Raushan_Kumar_IVP_Outputs",help="Output directory.")
    args=parser.parse_args()
    root=Path(args.output).resolve(); ensure_dirs(root)
    if args.video:
        video=Path(args.video).resolve()
        if not video.exists(): raise FileNotFoundError(video)
        input_note="User-provided input video"
    else:
        video=root/"RoadTraffic_Synthetic_Demo.mp4"
        make_synthetic_video(video)
        input_note="Synthetic demonstration video generated by this script; replace with faculty-approved input for final practical evidence."
    info=read_video_info(video)
    first=read_first_frame(video)
    metrics={"student":STUDENT,"registration_number":REG_NO,"subject_code":"21CSE480T",
             "input_video":str(video),"input_note":input_note,"video_metadata":info}
    metrics["p1_edge_detection"]=p1_edges(first,root/"output_images")
    metrics["p2_compression"]=p2_compression(first,root/"output_images")
    metrics["p3_segmentation"]=p3_segmentation(first,root/"output_images")
    metrics["p4_frame_extraction"]=p4_extract(video,root,stride=20)
    metrics["p5_video_sampling"]=p5_sampling(video,root)
    metrics["p6_geometric_transformations"]=p6_transform(first,root/"output_images")
    metrics["p7_frame_difference"]=p7_difference(video,root/"output_images")
    metrics["p8_spatial_filtering"]=p8_filter_video(video,root)
    metrics["p9_video_compression"]=p9_compress(video,root)
    metrics["p10_bitrate_streaming"]=p10_bitrate(video,root)
    save_json(root/"metrics"/"metrics_comparison.json",metrics)
    # CSV summary for core numeric metrics
    with (root/"metrics"/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["Problem","Metric","Value"])
        w.writerow(["P4","frames_saved",metrics["p4_frame_extraction"]["frames_saved"]])
        w.writerow(["P7","changed_pixel_percent",metrics["p7_frame_difference"]["changed_pixel_percent"]])
        w.writerow(["P8","mean_psnr_db",metrics["p8_spatial_filtering"]["psnr_db_mean"]])
        for row in metrics["p2_compression"]: w.writerow(["P2",row["method"],row["psnr_db"]])
    print(json.dumps(metrics,indent=2,default=str))
    print(f"\nDone. Outputs saved in: {root}")
    print("NOTE: If synthetic input was used, replace it with your actual input video and rerun before submission.")

if __name__=="__main__":
    main()
