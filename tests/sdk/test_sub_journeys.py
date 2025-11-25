# # Copyright 2025 Emcie Co Ltd.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from parlant.core.journeys import JourneyStore
# from parlant.core.services.tools.plugins import tool
# from parlant.core.tools import ToolContext, ToolResult
# from tests.sdk.utils import Context, SDKTest
# from tests.test_utilities import nlp_test

# from parlant import sdk as p


# class Test_linking_journey_mid_journey_and_back_with_fast_forwarding(SDKTest):
#     """Test room booking journey that links to user validation and handles validated/not validated conditions."""

#     async def setup(self, server: p.Server) -> None:
#         self.agent = await server.create_agent(
#             name="Hotel Booking Agent",
#             description="Help customers book hotel rooms with user validation",
#         )

#         self.validation_journey = await self.agent.create_journey(
#             title="User Validation Process",
#             conditions=["User needs to be validated"],
#             description="Collect and validate user information",
#         )

#         @tool
#         def validate_user_info(context: ToolContext, first_name: str) -> ToolResult:
#             return ToolResult(
#                 data={
#                     "validated": True,
#                 }
#             )

#         @tool
#         def book_room(context: ToolContext, room_type: str, user_id: str) -> ToolResult:
#             return ToolResult(
#                 data={
#                     "booking_confirmed": True,
#                     "booking_id": f"ROOM_{user_id}_{room_type}_001",
#                     "message": f"Room {room_type} booked successfully for user {user_id}",
#                 }
#             )

#         self.collect_names = await self.validation_journey.initial_state.transition_to(
#             chat_state="Please provide your first name",
#         )

#         self.validate_info = await self.collect_names.target.transition_to(
#             tool_instruction="Validate the provided user information",
#             tool_state=validate_user_info,
#         )

#         self.main_journey = await self.agent.create_journey(
#             title="Hotel Room Booking",
#             conditions=["Customer wants to book a room"],
#             description="Main hotel room booking process",
#         )

#         self.room_inquiry = await self.main_journey.initial_state.transition_to(
#             chat_state="What type of room would you like to book? the blue or the red one?",
#         )

#         self.need_validation = await self.room_inquiry.target.transition_to(
#             condition="Customer specifies room type and wants to book",
#             chat_state="To proceed with the booking, I need to validate your identity first.",
#         )

#         self.validation_link = await self.need_validation.target.transition_to(
#             journey=self.validation_journey,
#         )

#         self.validated_booking = await self.validation_link.target.transition_to(
#             condition="User is validated",
#             tool_instruction="Book the room for the validated user",
#             tool_state=book_room,
#         )

#         self.booking_confirmation = await self.validated_booking.target.transition_to(
#             condition="Booking completed successfully",
#             chat_state="Great! Your room has been booked successfully. You will receive a confirmation email shortly.",
#         )

#         self.not_validated_response = await self.validation_link.target.transition_to(
#             condition="User is not validated",
#             chat_state="I'm sorry, but I cannot proceed with the booking as we couldn't validate your information. Please contact customer service for assistance.",
#         )

#     async def run(self, ctx: Context) -> None:
#         journey_store = ctx.container[JourneyStore]

#         main_nodes = await journey_store.list_nodes(journey_id=self.main_journey.id)

#         validation_embedded = [
#             node
#             for node in main_nodes
#             if node.metadata.get("sub_journey_id") == self.validation_journey.id
#         ]

#         assert len(validation_embedded) >= 1, "Validation journey states should be embedded"

#         # Test successful validation and booking scenario
#         session_ctx = {"reuse_session": True}

#         # Start booking process
#         response1 = await ctx.send_and_receive(
#             "I want to book a hotel room", recipient=self.agent, **session_ctx
#         )

#         assert await nlp_test(context=response1, condition="Agent asks about room type preferences")

#         # Specify room type
#         response2 = await ctx.send_and_receive(
#             "I'd like to book a deluxe suite", recipient=self.agent, **session_ctx
#         )

#         assert await nlp_test(
#             context=response2, condition="Agent mentions needing validation and asks for name"
#         )

#         # Provide valid user information
#         response3 = await ctx.send_and_receive(
#             "My name is John Smith", recipient=self.agent, **session_ctx
#         )

#         # Should validate the user and confirm validation success
#         assert await nlp_test(
#             context=response3, condition="Agent confirms validation was successful"
#         )

#         # Continue with booking after validation
#         response4 = await ctx.send_and_receive(
#             "Please proceed with booking the deluxe suite", recipient=self.agent, **session_ctx
#         )

#         # Should book the room and confirm
#         assert await nlp_test(
#             context=response4, condition="Agent confirms room booking was successful"
#         )
