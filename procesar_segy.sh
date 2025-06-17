#!/bin/bash

# Ruta a los binarios de Seismic Unix
export PATH=$PATH:/Users/alejandravesga2024/SeisUnix/bin
export DYLD_LIBRARY_PATH=/Users/alejandravesga2024/SeisUnix/lib

# Argumentos del script
INPUT="$1"  # ruta al .sgy recibido por la app
BASENAME=$(basename "$INPUT" .sgy)
OUTPUT="output/procesado_${BASENAME}.su"
NS=2500  # número de muestras por traza (ajustado previamente)

# Convertir a SU
segyread tape="$INPUT" conv=1 > output/tmp_raw.su

# Corregir encabezado NS
sushw < output/tmp_raw.su key=ns a=$NS > output/tmp_ns.su

# Ganancia y filtro
sugain < output/tmp_ns.su tpow=2 | sufilter f=5,10,100,120 > "$OUTPUT"

# Convertir a SEG-Y
segywrite < "$OUTPUT" tape="output/${BASENAME}_procesado.sgy"

# Limpiar temporales
rm output/tmp_raw.su output/tmp_ns.su

# Salida
echo "output/${BASENAME}_procesado.sgy"


