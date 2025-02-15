import React from 'react';

interface Props {
    setFiles: React.Dispatch<React.SetStateAction<File[]>>,
    event: React.ChangeEvent<HTMLInputElement>   
}

export function guardar({ setFiles, event }: Props) {
    // si existen los files, iterar por los files
    const file = event.target.files?.[0];
    if (file) {
        // setFiles(file); este error da porque tienes que ir guardando los archivos en el bucle
    }
}