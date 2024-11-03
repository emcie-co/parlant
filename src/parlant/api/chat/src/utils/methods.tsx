import React from 'react';

export const spaceClick = (e: React.KeyboardEvent<HTMLElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') (e.target as HTMLElement).click();
};