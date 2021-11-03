from infertrade.utilities.api_automation import execute_it_api_request
import os


InsertYourApiKeyHere = os.environ.get('API_KEY')

response = execute_it_api_request(request_name="Get trading rule metadata",
                        api_key="myapi")

print(response.text)




















from infertrade.utilities.api_automation import execute_it_api_request
additional_data = {"price":[0,1,2,3,4,5,6,7,8,9],"research_1":[0,1,2,3,4,5,6,7,8,9]}
execute_it_api_request( request_name="Get available time series simulation models",
                        api_key = InsertYourApiKeyHere,
                        additional_data = additional_data)


from infertrade.utilities.api_automation import execute_it_api_request
execute_it_api_request( request_name="Get available time series simulation models",
                        api_key="YourApiKey",
                        request_body = "YourRequestBody",
                        header = "YourHeader")


from infertrade.utilities.api_automation import execute_it_api_request, parse_csv_file
data = parse_csv_file(file_name="File_Name")
additional = {"trailing_stop_loss_maximum_daily_loss": "value",
            "price": data["Column_Name"],
            "research_1": data["Column_Name"]}
response = execute_it_api_request(
            request_name="Get available time series simulation models",
            api_key="YourApiKey",
            additional_data=additional,
             )
print(response.txt)


