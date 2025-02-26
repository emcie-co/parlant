import {twJoin} from 'tailwind-merge';
import Tooltip from '../../../ui/custom/tooltip';
import {Copy, Fullscreen, X} from 'lucide-react';
import {copy} from '@/lib/utils';
import clsx from 'clsx';
import {Log} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {dialogAtom} from '@/store';
import {useRef} from 'react';

const MessageLog = ({log}: {log: Log}) => {
	const [dialog] = useAtom(dialogAtom);
	const ref = useRef<HTMLPreElement>(null);

	const openLogs = (text: string) => {
		const element = (
			<pre ref={ref} className='group px-[30px] py-[10px] text-wrap text-[#333] relative overflow-auto h-[100%]'>
				<div className='invisble group-hover:visible flex sticky top-[10px] right-[20px] justify-end'>
					<div className='flex justify-end bg-white p-[10px] gap-[20px] rounded-lg'>
						<Tooltip value='Copy' side='top'>
							<Copy onClick={() => copy(text, ref?.current || undefined)} size={18} className='cursor-pointer' />
						</Tooltip>
						<Tooltip value='Close' side='top'>
							<X onClick={() => dialog.closeDialog()} size={18} className='cursor-pointer' />
						</Tooltip>
					</div>
				</div>
				<div>{text}</div>
			</pre>
		);
		dialog.openDialog('', element, {height: '90vh', width: '90vw'});
	};

	return (
		<div className={twJoin('flex max-h-[100px] overflow-hidden group relative font-ubuntu-mono rounded-[8px] gap-[5px] px-[20px] border-white border text-[14px] transition-all hover:border-[#EDEDED] hover:bg-[#F5F6F8]')}>
			<div className='absolute hidden z-10 group-hover:flex right-[10px] top-[10px] gap-[10px]'>
				<Tooltip value='Copy' side='top'>
					<Copy size={18} onClick={() => copy(log?.message || '')} className='cursor-pointer' />
				</Tooltip>
				<Tooltip value='Full screen' side='top'>
					<Fullscreen size={20} className='cursor-pointer' onClick={() => openLogs(log?.message || '')} />
				</Tooltip>
			</div>
			<pre className={clsx('max-w-[-webkit-fill-available] pe-[10px] text-wrap [mask-image:linear-gradient(to_bottom,white_60px,_transparent)]')}>
				{log?.level ? `[${log.level}]` : ''}
				{log?.message}
			</pre>
		</div>
	);
};

export default MessageLog;
