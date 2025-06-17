import os
import subprocess
import uuid

def process_segy_and_return_path(input_path):
    # Directorios
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Identificador único por archivo
    uid = uuid.uuid4().hex[:8]

    # Archivos de trabajo
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    su_output = os.path.join(output_dir, f"{base_name}_{uid}.su")
    sgy_output = os.path.join(output_dir, f"{base_name}_{uid}_processed.sgy")

    tmp_raw = os.path.join(output_dir, f"tmp_raw_{uid}.su")
    tmp_ns = os.path.join(output_dir, f"tmp_ns_{uid}.su")
    tmp_gain = os.path.join(output_dir, f"tmp_gain_{uid}.su")

    # Parámetros SU
    NS = 2500  # Esto podrías automatizar con segyhdrs, pero ahora es fijo
    DT = 4000

    # Rutas SU
    seisunix_bin = "/Users/alejandravesga2024/SeisUnix/bin"
    seisunix_lib = "/Users/alejandravesga2024/SeisUnix/lib"
    env = os.environ.copy()
    env["PATH"] = f"{env['PATH']}:{seisunix_bin}"
    env["DYLD_LIBRARY_PATH"] = seisunix_lib

    try:
        # 1. Convertir a .su
        subprocess.run(f"segyread tape={input_path} conv=1 > {tmp_raw}", shell=True, check=True, env=env)

        # 2. Corregir encabezado ns
        subprocess.run(f"sushw < {tmp_raw} key=ns a={NS} > {tmp_ns}", shell=True, check=True, env=env)

        # 3. Aplicar ganancia
        subprocess.run(f"sugain < {tmp_ns} tpow=2 > {tmp_gain}", shell=True, check=True, env=env)

        # 4. Filtro de frecuencia
        subprocess.run(f"sufilter < {tmp_gain} f=5,10,100,120 > {su_output}", shell=True, check=True, env=env)

        # 5. Convertir a SEG-Y
        subprocess.run(f"segywrite < {su_output} tape={sgy_output}", shell=True, check=True, env=env)

        return sgy_output

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al ejecutar un comando de SU: {e}")

    finally:
        # Limpiar archivos temporales
        for tmp in [tmp_raw, tmp_ns, tmp_gain, su_output]:
            if os.path.exists(tmp):
                os.remove(tmp)
