import React from 'react';
import { SetFile } from '../types/State';

interface Props {
    setFiles: SetFile,
    event: React.ChangeEvent<HTMLInputElement>   
}

export function guardar({ setFiles, event }: Props) {
    const files = event.target.files;
    if (files) {
        // Iterar sobre los archivos y guardarlos
        for (let i = 0; i < files?.length; i++) {
            const file = files[i];
            setFiles(prevFiles => [...prevFiles, file]);
        }
    }
}
