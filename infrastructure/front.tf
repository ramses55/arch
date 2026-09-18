locals{
	html_path = "../src/front.html"
}
resource "local_file" "html" {
	depends_on = [ yandex_api_gateway.gateway ]
	content = templatefile( local.html_path, {
		api-gateway-url = yandex_api_gateway.gateway.domain	
	})
	filename = "${path.module}/front.html"
	
}
