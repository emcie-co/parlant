import Avatar from '@/components/avatar/avatar';
import HeaderWrapper from '@/components/header-wrapper/header-wrapper';
import CopyText from '@/components/ui/custom/copy-text';
import {agentAtom, customerAtom, sessionAtom} from '@/store';
import {AgentInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {memo} from 'react';

const SessoinViewHeader = () => {
	const [session] = useAtom(sessionAtom);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);

	return (
		<HeaderWrapper>
			{session?.id && (
				<div className='w-full flex items-center h-full pb-[2px] max-w-[1000px] m-auto'>
					<div className='h-full flex-1 flex items-center border-e border-[#F3F5F9] whitespace-nowrap overflow-hidden'>
						{agent && <Avatar agent={agent as AgentInterface} tooltip={false} />}
						<div>
							<div>{agent?.name}</div>
							<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
								<CopyText preText='Agent ID:' text={` ${agent?.id}`} textToCopy={agent?.id} />
							</div>
						</div>
					</div>
					<div className='h-full flex-1 flex items-center ps-[14px] whitespace-nowrap overflow-hidden'>
						{customer && <Avatar agent={customer as AgentInterface} tooltip={false} />}
						<div>
							<div>{(customer?.id == 'guest' && 'Guest') || customer?.name}</div>
							<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
								<CopyText preText='Customer ID:' text={` ${customer?.id}`} textToCopy={customer?.id} />
							</div>
						</div>
					</div>
				</div>
			)}
		</HeaderWrapper>
	);
};
export default memo(SessoinViewHeader);
