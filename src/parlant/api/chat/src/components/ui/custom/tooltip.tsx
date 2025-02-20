import {Tooltip as ShadcnTooltip, TooltipContent, TooltipProvider, TooltipTrigger} from '@/components/ui/tooltip';
import {ReactElement} from 'react';
import {ClassNameValue, twMerge} from 'tailwind-merge';

interface Props {
	children: ReactElement;
	value: string | ReactElement;
	delayDuration?: number;
	style?: React.CSSProperties;
	side?: 'bottom' | 'top' | 'right' | 'left';
	className?: ClassNameValue;
	align?: 'center' | 'end' | 'start' | undefined;
}

export default function Tooltip({children, value, className, style = {}, side = 'bottom', align = 'center', delayDuration = 0}: Props) {
	return (
		<TooltipProvider>
			<ShadcnTooltip delayDuration={delayDuration}>
				<TooltipTrigger asChild>{children}</TooltipTrigger>
				<TooltipContent
					side={side}
					align={align}
					style={{boxShadow: 'none', ...style}}
					className={twMerge('left-[34px] h-[32px] text-[13px] font-normal font-inter rounded-[20px] border border-[#EBECF0] border-solid bg-white p-[5px_16px_7px_16px]', className)}>
					<div>{value}</div>
				</TooltipContent>
			</ShadcnTooltip>
		</TooltipProvider>
	);
}
