import { guardar } from '../utils/onSubmitImage'
import { SetFile } from '../types/State'
import UploadIcon from '../assets/icons/UploadIcon'

interface Props {
  background: "primary" | "secondary",
  element: "file" | "button",
  setFiles: SetFile,
  children: React.ReactNode,
  iconColor?: "white" | "primary" | "secondary",
  type?: "fulled" | "outline",
  onChange?: () => void
  onClick?: () => void
}

export default function Button({ 
  background, 
  element, 
  children, 
  type = "fulled", 
  iconColor = "white", 
  setFiles, 
  onClick, 
}: Props) {
  if(element === "file") {
    return (
      <>
        <label 
          htmlFor="input_images" 
          className={`${type === "fulled" ? `bg-${background} text-white hover:bg-${background}-light` : `bg-white border-${background} border-2 text-${background} hover:brightness-120 hover:border-${background}-light`} p-3 w-fit m-auto mt-10 rounded-2xl flex justify-center text-left max-h-16 items-center gap-3 cursor-pointer transition duration-200 active:scale-90`}
        >
          <UploadIcon color={iconColor} />
          <p>{children}</p>
        </label>
        <input 
          id='input_images'
          type='file'  
          className='hidden'
          accept="image/*"
          multiple
          onChange={(event) => guardar({ setFiles, event })}
        />
      </>
    )
  } else {
    return (
      <button  
        className={`${type === "fulled" ? `bg-${background} text-white hover:bg-${background}-light` : `bg-white border-${background} border-2 text-${background} hover:brightness-110 hover:border-${background}-light`} p-3 w-fit m-auto mt-10 rounded-2xl flex justify-center text-left max-h-16 items-center gap-3 cursor-pointer transition duration-200 active:scale-90`}
        onClick={onClick}
      >
        { children }
      </button>
    )
  }
}
