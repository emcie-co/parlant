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

const ChatHeader = ({setFilterSessionVal, filterSessionVal}: {setFilterSessionVal: (value: string) => void; filterSessionVal: string}): ReactNode => {
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
				<div className='flex items-center min-[801px]:hidden'>
				<div className='flex items-center'>
						<img src='/chat/app-logo.svg' alt='logo' aria-hidden className='self-center h-[30px]' />
					</div>
					<div>
						<Sheet open={sheetOpen} onOpenChange={() => setSheetOpen(!sheetOpen)}>
							<SheetTrigger asChild onClick={() => setSheetOpen(true)}>
								<Menu className='ms-[24px] cursor-pointer' />
							</SheetTrigger>
							<SheetContent side='left' className='w-fit p-0 [&>button[type=button]]:hidden'>
								<SheetHeader>
									<SheetTitle className='text-center'></SheetTitle>
									<SheetDescription />
								</SheetHeader>
								<div className='flex items-center px-[12px] flex-1 relative !shadow-main'>
									<img src='icons/search.svg' alt='' className='absolute left-[24px]' />
									<Input placeholder='Filter sessions' onChange={(e) => setFilterSessionVal(e.target.value)} className='!ring-0 !ring-offset-0 h-[38px] w-full placeholder:font-light ps-[35px] rounded-[6px] !pointer-events-auto' />
								</div>
								<SessionList filterSessionVal={filterSessionVal} />
							</SheetContent>
						</Sheet>
					</div>
				</div>
				<a href='https://parlant.io' target='_blank' className='flex items-center ms-[4px] -me-[6px] max-mobile:hidden'>
					<img src='/chat/app-logo.svg' alt='logo' aria-hidden className='self-center h-[30px]' />
				</a>
				<div className='flex items-center ps-[12px] flex-1 relative !shadow-main max-mobile:hidden'>
					<img src='icons/search.svg' alt='' className='absolute left-[24px]' />
					<Input placeholder='Filter sessions' onChange={(e) => setFilterSessionVal(e.target.value)} className='!ring-0 !ring-offset-0 h-[38px] w-full placeholder:font-light ps-[35px] rounded-[6px] !pointer-events-auto' />
				</div>
				<div className='group ms-[8px]'>
					<Tooltip value='New Session' side='right' className='group'>
						<>
							<img src='buttons/new-session.svg' alt='add session' className='shadow-main cursor-pointer group-hover:hidden' tabIndex={1} role='button' onKeyDown={spaceClick} onClick={createNewSession} />
							<img src='buttons/new-session-hover.svg' alt='add session' className='shadow-main cursor-pointer hidden group-hover:block' tabIndex={1} role='button' onKeyDown={spaceClick} onClick={createNewSession} />
						</>
					</Tooltip>
				</div>
			</div>
			{/* <div className='w-[352px] h-[70px] flex items-center justify-end me-4'>
                <DarkModeToggle/>
            </div> */}
		</HeaderWrapper>
	);
};

export default ChatHeader;
