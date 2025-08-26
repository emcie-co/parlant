/* eslint-disable react-refresh/only-export-components */
import {AgentInterface, CustomerInterface} from '@/utils/interfaces';
import React, {ReactNode} from 'react';
import Tooltip from '../ui/custom/tooltip';
import {twMerge} from 'tailwind-merge';

interface Props {
	agent: AgentInterface;
	customer?: CustomerInterface;
	tooltip?: boolean;
}

interface Color {
	text: string;
	background: string;
	outerBackground: string;
	iconBackground?: string;
	iconText?: string;
}

const colors = {
	green: {dark: 'rgb(80 130 1)', light: 'rgb(80 130 1 / 10%)', extraLight: 'rgb(80 130 1 / 5%)'},
	purple: {dark: 'rgb(85 1 104)', light: 'rgb(85 1 104 / 10%)', extraLight: 'rgb(85 1 104 / 5%)'},
	pink: {dark: 'rgb(155 3 95)', light: 'rgb(155 3 95 / 10%)', extraLight: 'rgb(155 3 95 / 5%)'},
	orange: {dark: 'rgb(183 99 0)', light: 'rgb(183 99 0 / 10%)', extraLight: 'rgb(183 99 0 / 5%)'},
	blue: {dark: 'rgb(46 128 108)', light: 'rgb(46 128 108 / 10%)', extraLight: 'rgb(46 128 108 / 5%)'},
};

const agentColors: Color[] = [
	{text: 'white', background: colors.green.dark, outerBackground: colors.green.light},
	{text: 'white', background: colors.purple.dark, outerBackground: colors.purple.light},
	{text: 'white', background: colors.pink.dark, outerBackground: colors.pink.light},
	{text: 'white', background: colors.orange.dark, outerBackground: colors.orange.light},
	{text: 'white', background: colors.blue.dark, outerBackground: colors.blue.light},
];
const customerColors: Color[] = [
	{iconBackground: colors.green.dark, background: colors.green.light, text: colors.green.dark, outerBackground: colors.green.extraLight},
	{iconBackground: colors.purple.dark, background: colors.purple.light, text: colors.purple.dark, outerBackground: colors.purple.extraLight},
	{iconBackground: colors.pink.dark, background: colors.pink.light, text: colors.pink.dark, outerBackground: colors.pink.extraLight},
	{iconBackground: colors.orange.dark, background: colors.orange.light, text: colors.orange.dark, outerBackground: colors.orange.extraLight},
	{iconBackground: colors.blue.dark, background: colors.blue.light, text: colors.blue.dark, outerBackground: colors.blue.extraLight},
];

export const getAvatarColor = (id: string, type: 'agent' | 'customer') => {
	const palette = type === 'agent' ? agentColors : customerColors;
	const hash = [...id].reduce((acc, char) => acc + char.charCodeAt(0), 0);
	return palette[hash % palette.length];
};

const Avatar = ({agent, customer, tooltip = true}: Props): ReactNode => {
	const agentColor = getAvatarColor(agent.id, 'agent');
	const customerColor = customer && getAvatarColor(customer.id, 'customer');
	const isAgentUnavailable = agent?.name === 'N/A';
	const isCustomerUnavailable = customer?.name === 'N/A';
	const agentFirstLetter = agent.name.replaceAll(/>|</g, '')[0].toUpperCase();
	const isGuest = customer?.id === 'guest' || agent?.id === 'guest';
	const customerFirstLetter = isGuest ? 'G' : customer?.name?.[0]?.toUpperCase();
	const style: React.CSSProperties = {transform: 'translateY(17px)', fontSize: '13px !important', fontWeight: 400, fontFamily: 'inter'};
	if (!tooltip) style.display = 'none';

	return (
		<Tooltip value={`${agent.name} / ${!customer?.name || isGuest ? 'Guest' : customer.name}`} side='right' style={style}>
			<div className='relative select-none'>
				<div className={twMerge('size-[44px] rounded-[8px] flex me-[14px] items-center justify-center', agent && customer && 'size-[38px]')} style={{background: agent && customer ? '' : agentColor.outerBackground}}>
					<div
						style={{background: agentColor.background, color: agentColor.text}}
						aria-label={'agent ' + agent.name}
						className={twMerge('size-[36px] rounded-[5px] flex items-center justify-center text-white text-[20px] font-semibold', isAgentUnavailable && 'text-[14px] !bg-gray-300')}>
						{isAgentUnavailable ? 'N/A' : agentFirstLetter}
					</div>
				</div>
				{agent && customer && (
					<div
						style={{background: customerColor?.iconBackground, color: 'white'}}
						aria-label={'customer ' + customer.name}
						className={twMerge('absolute me-[3px] border border-white size-[18px] rounded-[4px] flex items-center justify-center text-white text-[12px] font-normal -bottom-[3px] right-[1px] z-10', isCustomerUnavailable && 'text-[8px] !bg-gray-300')}>
						{isCustomerUnavailable ? 'N/A' : customerFirstLetter}
					</div>
				)}
			</div>
		</Tooltip>
	);
};

export default Avatar;
