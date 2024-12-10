import { AgentInterface, CustomerInterface } from '@/utils/interfaces';
import React, { ReactNode } from 'react';
import Tooltip from '../ui/custom/tooltip';

interface Props {
    agent: AgentInterface;
    customer?: CustomerInterface;
    tooltip?: boolean;
}

const colors = ['#B4E64A', '#FFB800', '#B965CC', '#87DAC6', '#FF68C3'];

const getAvatarColor = (agentId: string) => {
    const hash = [...agentId].reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return colors[hash % colors.length];
};

const AgentAvatar = ({agent, customer, tooltip = true}: Props): ReactNode => {
    const agentBackground = getAvatarColor(agent.id);
    const customerBackground = customer && getAvatarColor(customer.id);
    const agentFirstLetter = agent.name === '<guest>' ? 'G' : agent.name[0].toUpperCase();
    const isGuest = customer?.name === '<guest>';
    const customerFirstLetter = isGuest ? 'G' : customer?.name?.[0]?.toUpperCase();
    const style: React.CSSProperties = {transform: 'translateY(17px)', fontSize: '13px !important', fontWeight: 400, fontFamily: 'inter'};
    if (!tooltip) style.display = 'none';

    return (
        <Tooltip value={`${agent.name} / ${(!customer?.name || isGuest) ? 'Guest' : customer.name}`} side='right' style={style}>
            <div className='relative'>
                <div style={{background: agentBackground}} aria-label={'agent ' + agent.name} className={' me-[10px] size-[38px] rounded-full flex items-center justify-center text-white text-[20px] font-semibold'}>
                    {agentFirstLetter}
                </div>
                {customer &&
                <div style={{background: customerBackground}} aria-label={'customer ' + customer.name} className={'absolute me-[3px] size-[20px] rounded-full flex items-center justify-center text-white text-[12px] font-semibold border bottom-0 right-0 z-10'}>
                    {customerFirstLetter}
                </div>}
            </div>
        </Tooltip>
    );
};

export default AgentAvatar;