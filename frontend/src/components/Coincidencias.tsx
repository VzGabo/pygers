// // import { useEffect, useState} from "react"

// // export default function Concidencias() {
    
// //     const [sospechoso, setSospechoso] = useState([])
// //     useEffect(() => {
// //         fetch("http://localhost:8000/compare-faces")
// //         .then(res => res.json())
// //         .then(res => console.log(res))
// //     }, [])

// //     return (    
// //         <>
// //     <h1 className="text-2xl font-bold mb-6 text-center">Coincidencias</h1>
    
    
// //     </>
// //     )
// // }

// import { useEffect, useState } from "react";

// interface CompareFacesResponse {
//   match: boolean;
//   image: string;
// }

// export default function Concidencias() {
//   const [matchedImage, setMatchedImage] = useState<string | null>(null);
//   const [loading, setLoading] = useState<boolean>(true);
//   const [error, setError] = useState<string | null>(null);

//   useEffect(() => {
//     fetch("http://localhost:8000/compare-faces")
//       .then((res) => {
//         if (!res.ok) {
//           throw new Error("Error en la solicitud");
//         }
//         return res.json() as Promise<CompareFacesResponse>;
//       })
//       .then((res) => {
//         setLoading(false);
//         if (res.match) {
//           setMatchedImage(res.image);
//         } else {
//           setMatchedImage(null);
//         }
//       })
//       .catch((err) => {
//         setLoading(false);
//         setError(err.message);
//       });
//   }, []);

//   return (
//     <>
//       <h1 className="text-4xl mb-8 flex flex-col items-center justify-center text-white font-bold mt-4">Coincidencias</h1>
//       {loading && (
//         <div className="flex justify-center items-center text-white">
//           <p>Cargando...</p>
//         </div>
//       )}
//       {error && (
//         <div className="flex justify-center items-center text-red-500">
//           <p>Error: {error}</p>
//         </div>
//       )}
//       {matchedImage && (
//         <div className="flex justify-center items-center">
//           <img
//             className="w-34 m-2 rounded border-4 border-orange-500"
//             src={matchedImage}
//             alt="Imagen coincidencia"
//           />
//         </div>
//       )}
//     </>
//   );
// }
