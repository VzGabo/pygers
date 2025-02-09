import Navbar from './components/Navbar'
import Button from './components/Button'

export default function App() {
  return (
    <main>
      <Navbar />
      <main className='flex justify-center mt-10'>
        <Button background='primary'>
          Sube tus sospechosos
        </Button>
      </main>
    </main>
  )
}
