import { spaceClick } from '@/utils/methods';
import styles from './gradient-button.module.scss';
import { ReactElement, ReactNode } from 'react';

interface GradientButtonProps {
  className?: string;
  buttonClassName?: string;
  children: ReactNode
  onClick: (e: React.MouseEvent) => void;
}

export default function GradientButton({className, buttonClassName, children, onClick}: GradientButtonProps): ReactElement {
  return (
    <span tabIndex={0} onKeyDown={spaceClick} onClick={onClick} data-testid="gradient-button" role='button' className={styles.colorsButton + ' relative block rounded-md border-2 border-transparent hover:animate-background-shift ' + (className || '')}>
      <div style={{backgroundSize: '200% 200%'}} className='z-0 absolute top-[-1px] bottom-[-1px] left-[-1px] right-[-1px] animate-background-shift blur-[4px]'></div>
      <span className={buttonClassName + ' ' + styles.children + ' button relative text-center flex justify-center'}>
          {children}
      </span>
    </span>
  );
}