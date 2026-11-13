#!/usr/bin/env python3
import subprocess, time, xml.etree.ElementTree as ET, sys, os

PKG = "com.romangold.tank2aivideostudio.debug"
SERIAL = "emulator-5554"

def adb(*args, check=True):
    p = subprocess.run(["adb", "-s", SERIAL, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and p.returncode:
        raise RuntimeError(f"adb {' '.join(args)} failed ({p.returncode}):\n{p.stdout}")
    return p.stdout.strip()

def snapshot():
    adb("shell", "uiautomator", "dump", "/sdcard/window.xml", check=False)
    raw = adb("exec-out", "cat", "/sdcard/window.xml", check=False)
    try:
        root = ET.fromstring(raw)
    except Exception:
        return []
    result = []
    for n in root.iter("node"):
        result.append({
            "text": n.attrib.get("text", ""),
            "desc": n.attrib.get("content-desc", ""),
            "class": n.attrib.get("class", ""),
            "bounds": n.attrib.get("bounds", ""),
            "clickable": n.attrib.get("clickable", "false")
        })
    return result

def center(bounds):
    import re
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
    if not m: return None
    x1,y1,x2,y2 = map(int,m.groups())
    return ((x1+x2)//2,(y1+y2)//2,(x2-x1)*(y2-y1))

def tap_node(node):
    c = center(node["bounds"])
    if not c: return False
    adb("shell","input","tap",str(c[0]),str(c[1]))
    return True

def dump_state(label):
    nodes = snapshot()
    print(f"\n=== UI {label} ===", flush=True)
    for n in nodes:
        if n["text"] or n["desc"]:
            print(f'text={n["text"]!r} desc={n["desc"]!r} class={n["class"]} bounds={n["bounds"]} clickable={n["clickable"]}', flush=True)
    return nodes

def wait_for(predicate, timeout, label):
    end = time.time()+timeout
    last = []
    while time.time()<end:
        last = snapshot()
        found = predicate(last)
        if found: return found, last
        time.sleep(2)
    dump_state(label+"_TIMEOUT")
    raise RuntimeError(f"Timed out waiting for {label}")

def click_text(nodes, needle, exact=False):
    for n in nodes:
        value=n["text"]
        if (value == needle if exact else needle.lower() in value.lower()) and n["bounds"]:
            if tap_node(n): return n
    return None

def main():
    image = os.environ.get("CI_IMAGE_PATH", "/sdcard/Pictures/ci-source.png")
    adb("push", "ci-artifacts/source-image.png", image)
    adb("shell","am","broadcast","-a","android.intent.action.MEDIA_SCANNER_SCAN_FILE","-d","file://"+image,check=False)
    time.sleep(4)
    print("=== INSTALLED PACKAGE ===", flush=True)
    print(adb("shell","pm","path",PKG), flush=True)
    adb("shell","monkey","-p",PKG,"1")
    nodes = wait_for(lambda ns: click_text(ns,"Выбрать изображение"), 90, "APP_HOME")
    node=click_text(nodes,"Выбрать изображение")
    if not node: raise RuntimeError("Cannot find image-picker button")
    dump_state("IMAGE_PICKER")
    time.sleep(3)
    nodes=snapshot()
    # Prefer the exact indexed photo if the picker exposes its filename.
    selected=click_text(nodes,"ci-source.png")
    if not selected:
        # Android photo picker/DocumentUI exposes the sole indexed image as a large image tile.
        candidates=[]
        for n in nodes:
            c=center(n["bounds"])
            if c and n["class"].endswith(("ImageView","View")) and c[2] > 90000:
                candidates.append((c[2],n))
        candidates.sort(reverse=True,key=lambda x:x[0])
        if candidates:
            selected=candidates[0][1]
            tap_node(selected)
        else:
            dump_state("IMAGE_TILE_NOT_FOUND")
            raise RuntimeError("Could not identify the indexed test photo in Android image picker")
    print("IMAGE_SELECTION_TAPPED", flush=True)
    time.sleep(5)
    nodes=dump_state("AFTER_IMAGE_SELECTION")
    # Wait for main screen and launch generation.
    gen, nodes = wait_for(lambda ns: click_text(ns,"Создать видео"), 60, "GENERATION_BUTTON")
    # The first run may present a non-cancelable license acceptance dialog.
    license_node=click_text(nodes,"Я прочитал(а) и принимаю")
    if license_node:
        print("LICENSE_CHECKBOX_TAPPED", flush=True)
        time.sleep(1)
        nodes=dump_state("LICENSE_ACCEPTANCE")
        accept=click_text(nodes,"Принять и продолжить")
        if not accept: raise RuntimeError("License checkbox was tapped but accept button not found")
        tap_node(accept)
        time.sleep(3)
        nodes=snapshot()
    gen=click_text(nodes,"Создать видео")
    if not gen:
        nodes=dump_state("GENERATION_BUTTON_MISSING")
        gen=click_text(nodes,"Создать видео")
    if not gen: raise RuntimeError("Generation button unavailable after image selection/license")
    tap_node(gen)
    print("GENERATION_BUTTON_TAPPED", flush=True)
    started=time.time()
    last_text=""
    timeout=5*60*60
    while time.time()-started < timeout:
        nodes=snapshot()
        texts=[n["text"] for n in nodes if n["text"]]
        joined=" | ".join(texts)
        if joined != last_text:
            print(f"UI_PROGRESS elapsed={int(time.time()-started)}s {joined}", flush=True)
            last_text=joined
        if "Готово" in joined:
            print("UI_GENERATION_SUCCESS", flush=True)
            break
        if "Ошибка:" in joined or "Генерация не завершена:" in joined:
            print("UI_GENERATION_ERROR "+joined, flush=True)
            raise RuntimeError("App reported generation error: "+joined)
        time.sleep(5)
    else:
        raise RuntimeError("Generation timed out after five hours")
    # Copy the app-private MP4 and summary to shared storage so Actions can collect them.
    adb("shell","run-as",PKG,"cp","files/generated/Tank2-MobileI2V.mp4","/sdcard/ci-generated.mp4")
    adb("pull","/sdcard/ci-generated.mp4","ci-artifacts/generated-video.mp4")
    adb("shell","run-as",PKG,"ls","-l","files/generated")
    print("VIDEO_BYTES="+str(os.path.getsize("ci-artifacts/generated-video.mp4")), flush=True)
    if os.path.getsize("ci-artifacts/generated-video.mp4") < 10000:
        raise RuntimeError("Generated MP4 is unexpectedly small")
    print("UI_AUTOMATION_SUCCESS", flush=True)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print("UI_AUTOMATION_FAILED: "+repr(e), flush=True)
        sys.exit(1)
