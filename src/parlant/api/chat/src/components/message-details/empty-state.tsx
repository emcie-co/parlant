import {ClassNameValue, twMerge} from 'tailwind-merge';

interface Props {
	title: string;
	subTitle?: string;
	className?: ClassNameValue;
	wrapperClassName?: ClassNameValue;
}

const EmptyState = ({title, subTitle, wrapperClassName, className}: Props) => {
	return (
		<div className={twMerge('flex flex-col m-auto justify-center items-center w-full h-full', wrapperClassName)}>
			<div className={twMerge('flex flex-col justify-center items-center -translate-y-[70px]', className)}>
				<img className='size-[224px] rounded-full' src='emcie-placeholder.svg' alt='' />
				<h2 className='text-[20px] font-medium font-inter text-[#656565] mt-[30px]'>{title}</h2>
				{subTitle && <p className='text-[15px] font-normal max-w-[378px] font-inter text-[#656565] text-center mt-[10px]'>{subTitle}</p>}
			</div>
		</div>
	);
};

export default EmptyState;
