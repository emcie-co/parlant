import {ReactNode, useEffect, useState} from 'react';
import Tooltip from '../ui/custom/tooltip';
import {spaceClick} from '@/utils/methods';
import AgentList from '../agents-list/agent-list';
import {Menu} from 'lucide-react';
import {Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger} from '../ui/sheet';
import SessionList from '../session-list/session-list';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useAtom} from 'jotai';
import {agentAtom, dialogAtom, sessionAtom} from '@/store';
import {Input} from '../ui/input';
// import DarkModeToggle from '../dark-mode-toggle/dark-mode-toggle';

export const NEW_SESSION_ID = 'NEW_SESSION';

const ChatHeader = ({setFilterSessionVal}: {setFilterSessionVal: any}): ReactNode => {
	const [sheetOpen, setSheetOpen] = useState(false);
	const [session, setSession] = useAtom(sessionAtom);
	const [, setAgent] = useAtom(agentAtom);
	const [dialog] = useAtom(dialogAtom);

	useEffect(() => {
		if (sheetOpen) setSheetOpen(false);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [session]);

	const createNewSession = () => {
		setSession(null);
		setAgent(null);
		dialog.openDialog('', <AgentList />, {height: '536px', width: '604px'});
	};

	return (
		<HeaderWrapper className='z-60 overflow-visible rounded-s-[16px] '>
			<div className='w-[352px] rounded-ss-[16px]  rounded-se-[16px] boder-b-[0.6px] border-b-[#ebecf0] max-mobile:w-full h-[70px] flex items-center max-mobile:justify-between bg-white'>
				<div className='flex items-center min-[751px]:hidden'>
					<div>
						<Sheet open={sheetOpen} onOpenChange={() => setSheetOpen(!sheetOpen)}>
							<SheetTrigger asChild onClick={() => setSheetOpen(true)}>
								<Menu className='ms-[24px] cursor-pointer' />
							</SheetTrigger>
							<SheetContent side='left' className='w-fit px-0'>
								<SheetHeader>
									<SheetTitle className='text-center'></SheetTitle>
									<SheetDescription />
								</SheetHeader>
								<SessionList filterSessionVal='' />
							</SheetContent>
						</Sheet>
					</div>
				</div>
				<div className='flex items-center ps-[12px] relative !shadow-main'>
					<img src='icons/search.svg' alt='' className='absolute left-[24px]' />
					<Input onChange={(e) => setFilterSessionVal(e.target.value)} className='!ring-0 !ring-offset-0 h-[38px] w-[250px] ps-[35px] rounded-[6px]' />
				</div>
				<div className='group ms-[8px]'>
					<Tooltip value='New Session' side='right' className='group'>
						{/* <div
							tabIndex={1}
							role='button'
							onKeyDown={spaceClick}
							onClick={createNewSession}
							className='hover:bg-[#F3F5F9] w-[70px] cursor-pointer py-[9px] h-[38px] ps-[12px] rounded-[10px] shadow-main */}
						{/* border hover:border-[#E9EBEF] pe-[8px] flex items-center gap-[4px]'> */}
						<>
							<img src='buttons/new-session.svg' alt='add session' className='cursor-pointer group-hover:hidden' tabIndex={1} role='button' onKeyDown={spaceClick} onClick={createNewSession} />
							<img src='buttons/new-session-hover.svg' alt='add session' className='cursor-pointer hidden group-hover:block' tabIndex={1} role='button' onKeyDown={spaceClick} onClick={createNewSession} />
						</>
						{/* <div className='font-medium text-[14px]'>New</div> */}
						{/* </div> */}
					</Tooltip>
				</div>
			</div>
			{/* <div className='flex-1 flex'>
				<div className='bg-red-300 flex-1'>A</div>
				<div className='bg-transparent flex-1'></div>
			</div> */}
			{/* <div className='w-[352px] h-[70px] flex items-center justify-end me-4'>
                <DarkModeToggle/>
            </div> */}
		</HeaderWrapper>
	);
};

export default ChatHeader;
