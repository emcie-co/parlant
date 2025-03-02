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

interface Color {
	text: string;
	background: string;
}

const agentColors: Color[] = [
	{text: '#649614', background: '#B4E64A1A'},
	{text: '#69157C', background: '#B965CC1A'},
	{text: '#AF1873', background: '#FF68C31A'},
	{text: '#CB7714', background: '#FFB8001A'},
	{text: '#419480', background: '#87DAC61A'},
];
const customerColors: Color[] = [
	{text: 'white', background: '#508200'},
	{text: 'white', background: '#550168'},
	{text: 'white', background: '#9B045F'},
	{text: 'white', background: '#B76300'},
	{text: 'white', background: '#2D806C'},
];

export const getAvatarColor = (id: string, type: 'agent' | 'customer') => {
	const palette = type === 'agent' ? agentColors : customerColors;
	const hash = [...id].reduce((acc, char) => acc + char.charCodeAt(0), 0);
	return palette[hash % palette.length];
};

const Agent = ({agent, customer, tooltip = true, asCustomer = false}: Props): ReactNode => {
	const agentColor = getAvatarColor(agent.id, asCustomer ? 'customer' : 'agent');
	const customerColor = customer && getAvatarColor(customer.id, 'customer');
	const isAgentUnavailable = agent?.name === 'N/A';
	const isCustomerUnavailable = customer?.name === 'N/A';
	const agentFirstLetter = agent.name === '<guest>' ? 'G' : agent.name[0].toUpperCase();
	const isGuest = customer?.name === '<guest>' || (asCustomer && agent.name === '<guest>');
	const customerFirstLetter = isGuest ? 'G' : customer?.name?.[0]?.toUpperCase();
	const style: React.CSSProperties = {transform: 'translateY(17px)', fontSize: '13px !important', fontWeight: 400, fontFamily: 'inter'};
	if (!tooltip) style.display = 'none';

	return (
		<Tooltip value={`${agent.name} / ${!customer?.name || isGuest ? 'Guest' : customer.name}`} side='right' style={style}>
			<div className='relative'>
				<div className='size-[44px] rounded-[6.5px] flex me-[10px] items-center justify-center' style={{background: agentColor.background}}>
					<div
						style={{background: customer ? '' : agentColor.background, color: asCustomer ? 'white' : agentColor.text}}
						aria-label={'agent ' + agent.name}
						className={twMerge('size-[36px] rounded-[5px] flex items-center justify-center text-white text-[20px] font-semibold', isAgentUnavailable && 'text-[14px] !bg-gray-300')}>
						{isAgentUnavailable ? 'N/A' : agentFirstLetter}
					</div>
				</div>
				{agent && customer && (
					<div
						style={{background: customerColor?.background, color: customerColor?.text}}
						aria-label={'customer ' + customer.name}
						className={twMerge('absolute me-[3px] size-[16px] rounded-[3.75px] flex items-center justify-center text-white text-[12px] font-semibold bottom-[2px] right-[5px] z-10', isCustomerUnavailable && 'text-[8px] !bg-gray-300')}>
						{isCustomerUnavailable ? 'N/A' : customerFirstLetter}
					</div>
				)}
			</div>
		</Tooltip>
	);
};

export default Agent;
