import logo from '../assets/logo.png'

export default function Navbar() {
  return (
    <nav className='flex items-center gap-4 p-5 bg-primary'>
      <h1 className="text-4xl text-white font-bold">
        CatchIt
      </h1>
      <img src={logo} alt="logo" className='w-16' />
    </nav>
  )
}
