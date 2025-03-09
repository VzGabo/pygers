import { Link } from 'react-router-dom'
import logo from '../assets/logo.png'

export default function Navbar() {
  return (
    <nav className='flex items-center gap-4 p-5 bg-primary'>
      <Link to={"/"} className='flex items-center gap-3'>
        <h1 className="text-4xl text-white font-bold">
          CatchIt
        </h1>
        <img src={logo} alt="logo" className='w-16' />
      </Link>
    </nav>
  )
}
