/* eslint-disable react-refresh/only-export-components */
import {AgentInterface, CustomerInterface} from '@/utils/interfaces';
import React, {ReactNode} from 'react';
import Tooltip from '../ui/custom/tooltip';
import {twMerge} from 'tailwind-merge';

interface Props {
	agent: AgentInterface;
	customer?: CustomerInterface;
	tooltip?: boolean;
	asCustomer?: boolean;
}

const colors = [
	{agentName: '#649614', customerName: '#508200', background: '#B4E64A1A'},
	{agentName: '#69157C', customerName: '#550168', background: '#B965CC1A'},
	{agentName: '#AF1873', customerName: '#9B045F', background: '#FF68C31A'},
	{agentName: '#CB7714', customerName: '#B76300', background: '#FFB8001A'},
	{agentName: '#419480', customerName: '#2D806C', background: '#87DAC61A'},
];

export const getAvatarColor = (agentId: string) => {
	const hash = [...agentId].reduce((acc, char) => acc + char.charCodeAt(0), 0);
	return colors[hash % colors.length];
};

const AgentAvatar = ({agent, customer, tooltip = true, asCustomer = false}: Props): ReactNode => {
	const agentColor = getAvatarColor(agent.id);
	const customerColor = customer && getAvatarColor(customer.id);
	const isAgentUnavailable = agent?.name === 'N/A';
	const isCustomerUnavailable = customer?.name === 'N/A';
	const agentFirstLetter = agent.name === '<guest>' ? 'G' : agent.name[0].toUpperCase();
	const isGuest = customer?.name === '<guest>';
	const customerFirstLetter = isGuest ? 'G' : customer?.name?.[0]?.toUpperCase();
	const style: React.CSSProperties = {transform: 'translateY(17px)', fontSize: '13px !important', fontWeight: 400, fontFamily: 'inter'};
	if (!tooltip) style.display = 'none';

	return (
		<Tooltip value={`${agent.name} / ${!customer?.name || isGuest ? 'Guest' : customer.name}`} side='right' style={style}>
			<div className='relative'>
				<div className='size-[44px] rounded-[6.5px] flex me-[10px] items-center justify-center' style={{background: agentColor.background}}>
					<div
						style={{background: customer ? '' : agentColor[asCustomer ? 'customerName' : 'background'], color: asCustomer ? 'white' : agentColor.agentName}}
						aria-label={'agent ' + agent.name}
						className={twMerge('size-[36px] rounded-[5px] flex items-center justify-center text-white text-[20px] font-semibold', isAgentUnavailable && 'text-[14px] !bg-gray-300')}>
						{isAgentUnavailable ? 'N/A' : agentFirstLetter}
					</div>
				</div>
				{customer && (
					<div
						style={{background: customerColor?.customerName, color: 'white'}}
						aria-label={'customer ' + customer.name}
						className={twMerge('absolute me-[3px] size-[16px] rounded-[3.75px] flex items-center justify-center text-white text-[12px] font-semibold bottom-[2px] right-[5px] z-10', isCustomerUnavailable && 'text-[8px] !bg-gray-300')}>
						{isCustomerUnavailable ? 'N/A' : customerFirstLetter}
					</div>
				)}
			</div>
		</Tooltip>
	);
};

export default AgentAvatar;
