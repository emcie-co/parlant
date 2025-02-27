import AgentAvatar from '@/components/agent-avatar/agent-avatar';
import HeaderWrapper from '@/components/header-wrapper/header-wrapper';
import CopyText from '@/components/ui/custom/copy-text';
import {agentAtom, customerAtom, sessionAtom} from '@/store';
import {AgentInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {twJoin} from 'tailwind-merge';

const SessoinViewHeader = () => {
	const [session] = useAtom(sessionAtom);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	return (
		<HeaderWrapper className={twJoin('border-e')}>
			{session?.id && (
				<div className='w-full flex items-center h-full'>
					<div className='h-full flex-1 flex items-center ps-[24px] border-e'>
						<AgentAvatar agent={agent as AgentInterface} tooltip={false} />
						<div>
							<div>{agent?.name}</div>
							<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
								<CopyText preText='Agent ID:' text={` ${agent?.id}`} textToCopy={agent?.id} />
							</div>
						</div>
					</div>
					<div className='h-full flex-1 flex items-center ps-[24px]'>
						<AgentAvatar agent={customer as AgentInterface} asCustomer tooltip={false} />
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
export default SessoinViewHeader;
