import {ReactNode} from 'react';
import {twMerge} from 'tailwind-merge';
import Spacer from '../ui/custom/spacer';

const HeaderWrapper = ({children, className}: {children?: ReactNode; className?: string}) => {
	return <div className={twMerge('h-[70px] bg-white min-h-[70px] rounded-se-[16px] border-[#F3F5F9] rounded-ss-[16px] flex justify-between border-b-[0.6px] border-b-solid sticky top-0 z-10', className)}>
		<Spacer/>
		{children}
		<Spacer/>
		</div>;
};

export default HeaderWrapper;
