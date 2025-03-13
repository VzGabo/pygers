// import { useEffect } from 'react';

// interface ImageTransferProps {
//   file: File; // Archivo de imagen
//   onTransfer: (url: string) => void; // Función para transferir la URL al componente padre
// }

// const ImageTransfer = ({ file, onTransfer }: ImageTransferProps) => {
//   useEffect(() => {
//     // Crear la URL temporal
//     const objectURL = URL.createObjectURL(file);

//     // Transferir la URL al componente padre
//     onTransfer(objectURL);

//     // Limpiar la URL cuando el componente se desmonte
//     return () => {
//       URL.revokeObjectURL(objectURL);
//     };
//   }, [file, onTransfer]);

//   return null; // Este componente no renderiza nada
// };

// export default ImageTransfer;