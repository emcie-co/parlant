import React from 'react';

export const spaceClick = (e: React.KeyboardEvent<HTMLElement>): void => {
	if (e.key === 'Enter' || e.key === ' ') (e.target as HTMLElement).click();
};

export function getDistanceToRight(element: HTMLElement): number {
	const rect = element.getBoundingClientRect();
	const distanceToRight = window.innerWidth - rect.right;
	return distanceToRight;
}
