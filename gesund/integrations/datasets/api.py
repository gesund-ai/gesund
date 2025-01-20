# An API file to interface between user and the dataset files
from _authentication import AuthService

class DatasetAPIManager:
    def get_dataset_list():
        """Function to get list of datasets."""
        base_url = "http://localhost:8123"
        endpoint = "/dataset"
        user_name = "user"
        password = "password"

        AuthService.get_auth_token(user_name, password, base_url, "gesund")

        response = AuthService.make_authenticated_request(
            endpoint=endpoint,
            method="GET",
            user_name=user_name,
            password=password,
            base_url=base_url,
            keyring_service_name="gesund"
        )

        return response
    

def main():
    # Initialize DatasetAPIManager and get dataset list
    dataset_list = DatasetAPIManager.get_dataset_list()
    print("Dataset List:", dataset_list)

if __name__ == "__main__":
    main()