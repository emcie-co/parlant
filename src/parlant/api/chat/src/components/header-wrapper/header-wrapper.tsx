import {ReactNode} from 'react';
import {twMerge} from 'tailwind-merge';

const HeaderWrapper = ({children, className}: {children?: ReactNode; className?: string}) => {
	return <div className={twMerge('h-[70px] bg-white min-h-[70px] rounded-se-[16px] border-[#F3F5F9] rounded-ss-[16px] flex justify-between sticky top-0 z-10', className)}>
		<div className='w-[12px] min-w-[12px]'/>
		{children}
		<div className='w-[12px] min-w-[12px]'/>
		</div>;
};

export default HeaderWrapper;
