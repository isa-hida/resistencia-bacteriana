"""Ejecuta y guarda las salidas del cuaderno en un kernel Python real."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/05_analisis_bayesiano.ipynb"
nb = nbformat.read(path, as_version=4)


def progreso(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Ejecutando celda {cell_index}...", flush=True)


client = NotebookClient(nb, timeout=180, startup_timeout=60, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=progreso)
client.execute()
nbformat.write(nb, path)
errores = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
           if o.output_type == "error"]
print("Celdas de codigo ejecutadas:", sum(c.cell_type == "code" for c in nb.cells))
print("Errores:", len(errores))
if errores:
    raise RuntimeError("El cuaderno contiene errores.")
