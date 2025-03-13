import Navbar from '../components/Navbar';
import { Link } from 'react-router-dom';

export default function Opciones() {
  return (
    <>
      <Navbar />
      <h1 className="text-4xl mb-8 flex flex-col items-center justify-center text-white font-bold mt-4">Seleccione su medio de búsqueda</h1>

      <div className="flex flex-col items-center space-y-4">
        <Link className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90" to="/Live">
          <button type="button">
            Iniciar cámara (captura en vivo)
          </button>
        </Link>

        <Link className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90 mb-4" to="/Grabacion">
          <button type="button">
            Subir video
          </button>
        </Link>

        <Link className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90" to="/">
          <button type="button">
            Cambiar sospechoso
          </button>
        </Link>
      </div>
    </>
  );
}
