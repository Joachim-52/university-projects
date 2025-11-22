# Terminal & Workflow Cheatsheet

## 1. Allgemeine Terminal-Basics
| Befehl | Beschreibung |
| :--- | :--- |
| `ls -la` | Listet alle Dateien (inkl. versteckte) mit Details auf |
| `cd <pfad>` | Wechselt das Verzeichnis (`cd ..` = zurück) |
| `mkdir <name>` | Erstellt einen neuen Ordner |
| `rm -rf <name>` | Löscht Dateien/Ordner **unwiderruflich** |
| `pwd` | Zeigt den aktuellen Pfad an |
| `history \| grep <xy>` | Sucht in der Befehlshistorie nach "xy" |

---

## 2. Git Workflow
* **Status prüfen:** `git status`
* **Staging:** `git add .` (Alles hinzufügen)
* **Speichern:** `git commit -m "Beschreibung"`
* **Hochladen:** `git push`
* **Aktualisieren:** `git pull`
* **Branching:** `git checkout -b <neuer-branch>`

---

## 3. Python Environments

| Aktion | venv (Standard) | Conda (Data Science) |
| :--- | :--- | :--- |
| **Erstellen** | `python -m venv .venv` | `conda create -n myenv python=3.9` |
| **Aktivieren (Mac/Linux)** | `source .venv/bin/activate` | `conda activate myenv` |
| **Aktivieren (Win)** | `.venv\Scripts\activate` | `conda activate myenv` |
| **Deaktivieren** | `deactivate` | `conda deactivate` |
| **Installieren** | `pip install <paket>` | `conda install <paket>` |
| **Exportieren** | `pip freeze > requirements.txt` | `conda env export > environment.yml` |

---

## 4. Speicherung im Repo (Best Practices)

### Option A: Makefile (für Schnellbefehle)
Erstelle eine Datei namens `Makefile` im Root-Verzeichnis:

```makefile
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	python main.py