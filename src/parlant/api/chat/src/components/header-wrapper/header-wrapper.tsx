import {ReactNode} from 'react';
import {twMerge} from 'tailwind-merge';

const HeaderWrapper = ({children, className}: {children?: ReactNode; className?: string}) => {
	return <div className={twMerge('h-[58px] bg-white min-h-[58px] rounded-se-[16px] rounded-ss-[16px] flex justify-between border-b-[0.6px] border-b-solid border-muted sticky top-0 z-10', className)}>{children}</div>;
};

export default HeaderWrapper;
